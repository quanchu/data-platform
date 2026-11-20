'use strict';

/** Split by Python character offsets, never by JavaScript UTF-16 code units. */
function sourceSegments(source, spans) {
    const characters = Array.from(source);
    const segments = [];
    const orderedSpans = spans.filter(span =>
        Number.isInteger(span.start) && Number.isInteger(span.end)
    ).sort((left, right) => left.start - right.start);
    let cursor = 0;
    for (const span of orderedSpans) {
        const start = Math.max(cursor, Math.min(characters.length, span.start));
        const end = Math.max(start, Math.min(characters.length, span.end));
        if (end <= start) continue;
        if (start > cursor) {
            segments.push({text: characters.slice(cursor, start).join(''), highlighted: false});
        }
        segments.push({text: characters.slice(start, end).join(''), highlighted: true});
        cursor = end;
    }
    if (cursor < characters.length) {
        segments.push({text: characters.slice(cursor).join(''), highlighted: false});
    }
    return segments;
}

/** Stable workflow phases; repeated model calls remain evidence within a phase. */
function workflowSteps(task) {
    const workflows = {
        segmentation: [
            ['structure', 'Read document structure', ['markdown_structure', 'heading_detection', 'input_block']],
            ['boundaries', 'Find topic boundaries', ['embedding_batch', 'gap_scores', 'section']],
            ['headings', 'Build section headings', ['heading_generation', 'block_headings']]
        ],
        claimify: [
            ['source', 'Read source sentences', ['source_parsing']],
            ['selection', 'Select factual content', ['selection', 'filter', 'filter_check']],
            ['clarification', 'Resolve references', ['ambiguity', 'clarify', 'clarify_check']],
            ['claims', 'Generate claims', ['generate']],
            ['validation', 'Validate claims and coverage', ['claim_check', 'coverage']]
        ],
        openie: [
            ['source', 'Read source sentences', ['source_parsing']],
            ['references', 'Resolve references', ['resolution']],
            ['relations', 'Extract relations', ['extraction']]
        ],
        entities: [
            ['model', 'Load entity model', ['model_loading']],
            ['candidates', 'Find entity candidates', ['entity_chunk']],
            ['mentions', 'Select final mentions', ['entity_selection']]
        ],
        claim_store: [
            ['save', 'Save unverified claims', ['store_unverified']],
            ['index', 'Prepare verified claim index', ['cache_sync', 'embedding_batch']],
            ['retrieve', 'Find related claims', ['retrieval']],
            ['compare', 'Compare claims', ['comparison']]
        ]
    };
    return [...(workflows[task] || []), ['result', 'Final result', ['run']]];
}

function workflowProgress(task, events) {
    const terminal = events.findLast(event => event.stage === 'run');
    const done = ['completed', 'failed', 'cancelled'].includes(terminal?.status);
    return workflowSteps(task).map(([id, title, stages]) => {
        const indices = events.flatMap((event, index) => stages.includes(event.stage) ? [index] : []);
        const latest = events[indices.at(-1)];
        let state = latest ? 'In progress' : done ? 'Not used' : 'Waiting';
        if (latest?.status === 'failed') state = 'Failed';
        else if (id === 'result') state = {completed: 'Complete', failed: 'Failed', cancelled: 'Stopped', cancelling: 'Stopping…', running: 'Waiting'}[terminal?.status] || 'Waiting';
        else if (done && latest) state = terminal.status === 'completed' ? 'Complete' : 'Stopped';
        return {id, title, indices, state};
    });
}
