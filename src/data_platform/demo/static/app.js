'use strict';
const $ = id => document.getElementById(id);
const tasks = [['segmentation', 'Segmentation'], ['claimify', 'Claimify'], ['openie', 'OpenIE'], ['entities', 'Entity tagging'], ['claim_store', 'Claim store']];
const descriptions = {
    input_block: ["Preserve an input block", "This source block is passed intact to the model. Lists and their introductions are protected from internal heading insertion."],
    markdown_structure: ['Recognize document structure', 'Identify headings, metadata, and complete lists before measuring topic changes. Protected blocks never enter prose scoring.'],
    gap_scores: ['Find changes in topic', 'Similarity measures neighboring prose windows. Depth measures a local drop relative to nearby peaks. Lists are excluded from both.'],
    section: ['Preserve a source section', 'This section is an exact slice of the article. List items and their nested content remain together.'],
    embedding_batch: ['Measure sentence meaning', 'Embed a bounded batch of text chunks, then combine chunks belonging to the same source sentence.'],
    heading_detection: ['Check an existing heading', 'Use the surrounding text to decide whether this candidate labels content or is ordinary body text.'],
    heading_generation: ['Describe the section', 'Generate a heading using the article context, without rewriting the original section.'],
    block_headings: ['Choose section headings', 'The model returns headings for indexed blocks. Complete lists remain atomic blocks.'],
    source_parsing: ['Locate source sentences', 'Read the H1 title and H2 sections; retain original sentence offsets and exclude code and tables.'],
    selection: ['Select factual content', 'Decide whether the sentence contains verifiable assertions. Verifiable does not mean independently verified.'],
    filter: ['Separate factual assertions', 'Remove subjective material while retaining the supported factual content.'],
    filter_check: ['Check the filtered sentence', 'Check support, factual coverage, and removal of subjective material.'],
    ambiguity: ['Check references and ambiguity', 'Find pronouns, abbreviated names, and other references that need explicit context.'],
    clarify: ['Make context explicit', 'Resolve supported references without inventing missing details.'],
    clarify_check: ['Validate clarification', 'Check that the rewrite preserves the source meaning and resolves ambiguity.'],
    generate: ['Extract individual claims', 'Turn the clarified sentence into independently understandable factual claims.'],
    claim_check: ['Validate each claim', 'Check entailment, verifiability, and sufficient context against the original sentence.'],
    coverage: ['Check factual coverage', 'Ensure the claim set retains all verifiable content. Failed criteria can trigger a repair attempt.'],
    resolution: ['Resolve section references', 'Identify references within the section and record evidence for replacements.'],
    extraction: ['Extract relation tuples', 'Extract subjects, relations, objects, and qualifiers from the resolved sentence.'],
    model_loading: ['Load the local entity model', 'Load weights into CPU memory. Subsequent runs reuse the loaded model.'],
    entity_chunk: ['Find entity candidates', 'Run inference on an overlapping source chunk and map mentions back to the article.'],
    entity_selection: ['Select entity mentions', 'Keep the highest-scoring duplicates and resolve conflicting overlaps.'],
    store_unverified: ['Save for human review', 'Save a source-aligned claim as unverified. This action never approves a claim.'],
    cache_sync: ['Synchronize verified claims', 'Reuse valid cached vectors and embed changed verified records. JSON files remain the source of truth.'],
    retrieval: ['Retrieve related verified claims', 'Rank verified records by embedding similarity. Retrieval is a candidate search, not a truth judgment.'],
    comparison: ['Compare factual assertions', 'Compare scope, dates, units, attribution, and meaning against retrieved claims.'],
    run: ['Processing run', 'Events come directly from the running task. Select a completed step to inspect its evidence.']
};
let stopping = false;
let sourceView = 'raw';
let task = 'segmentation', source = '', sourceName = 'article.md', events = [], selected = -1, currentRun = null, result = null, stream = null, busy = false, lastClaimify = null;
function element(tag, text, className) {
    const el = document.createElement(tag);
    if (text !== undefined)
        el.textContent = text;
    if (className)
        el.className = className;
    return el;
}
async function api(path, options) {
    const response = await fetch(path, options);
    const body = await response.json();
    if (!response.ok)
        throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body));
    return body;
}
function status(text, error = false) {
    $('status').textContent = text;
    $('status').classList.toggle('error', error);
}
function setBusy(value) {
    busy = value;
    $('run').disabled = value;
    $('cancel').disabled = !value || !currentRun || stopping;
    for (const el of document.querySelectorAll('#tabs button,.input-tools button,.input-tools select,.input-tools input,#apply,.settings input,.settings select'))
        el.disabled = value;
}
function resetView() {
    if (stream) stream.close();
    stream = null;
    events = [];
    stopping = false;
    $('cancel').textContent = 'Stop task';
    selected = -1;
    result = null;
    currentRun = null;
    buildSteps();
    $('cards').replaceChildren();
    $('chart').replaceChildren();
    $('handoffs').replaceChildren();
    $('details').textContent = '';
    $('event-count').textContent = '0 steps';
    $('download').disabled = true;
    $('final-panel').hidden = true;
    $('view-final').hidden = true;
    $('save-status').textContent = '';
    $('save-result').disabled = false;
    summary('Ready to explore', 'Choose settings, then run the task on the current source.');
    highlight([]);
    status('Ready to explore');
}
function selectTask(value) {
    if (busy)
        return;
    task = value;
    resetView();
    for (const button of $('tabs').children)
        button.classList.toggle('active', button.dataset.task === task);
    $('run').replaceChildren(document.createTextNode('Run ' + tasks.find(item => item[0] === task)[1].toLowerCase()), element('span', '↗'));
    for (const el of document.querySelectorAll('[data-tasks]'))
        el.hidden = !el.dataset.tasks.split(' ').includes(task);
    $('store-panel').hidden = task !== 'claim_store';
    $('settings-hint').textContent = task === 'segmentation' ? 'Live segmentation · configurable models' : task === 'entities' ? 'Local model inference' : task === 'claim_store' ? 'Manual approval remains separate' : 'Live model calls · server credentials';
    if (task === 'claim_store')
        refreshClaims();
}
function setSource(text, name) {
    source = text;
    sourceName = name;
    $('editor').value = text;
    $('source-preview').replaceChildren();
    setSourceView(sourceView);
    $('source-name').textContent = name;
    $('char-count').textContent = Array.from(text).length.toLocaleString() + ' chars';
    $('source-kind').textContent = name.includes('segmented') ? 'Generated Markdown' : 'Original Markdown';
    resetView();
}
// Python offsets count Unicode code points; JavaScript string indices count UTF-16 units.
function highlight(spans) {
    if (spans.length) setSourceView('raw');
    const fragment = document.createDocumentFragment();
    for (const segment of sourceSegments(source, spans)) {
        fragment.append(segment.highlighted
            ? element('mark', segment.text)
            : document.createTextNode(segment.text));
    }
    $('source').replaceChildren(fragment);
    const mark = $('source').querySelector('mark');
    if (mark) {
        $('source').scrollTop = Math.max(0, mark.offsetTop - $('source').offsetTop - 65);
    }
}
function summary(title, text) {
    $('stage-summary').replaceChildren(element('h3', title), element('p', text));
}
function card(title, text, spans = [], note = '') {
    const button = element('button', undefined, 'result-card');
    button.append(element('small', title), element('p', String(text)));
    if (note)
        button.append(element('div', note, 'source-label'));
    button.onclick = () => highlight(spans);
    $('cards').append(button);
}
function showCards(data) {
    $('cards').replaceChildren();
    if (!data)
        return;
    const output = data.output || data;
    if (output.heading || data.heading)
        card('Heading', output.heading || data.heading);
    for (const section of output.sections || [])
        card(section.kind, section.text, [section]);
    for (const entity of output.entities || [])
        card(entity.label + ' · ' + Number(entity.score).toFixed(2), entity.text, [entity]);
    for (const claim of output.claims || output.records || [])
        card('Claim', typeof claim === 'string' ? claim : claim.text, typeof claim === 'string' ? (events[selected]?.spans || []) : [claim]);
    for (const sentence of output.sentences || [])
        if (sentence.text)
            card(sentence.status || 'Sentence', sentence.text, Number.isInteger(sentence.start) ? [sentence] : []);
    for (const proposition of output.propositions || [])
        card('Relation', `${proposition.subject} → ${proposition.relation} → ${proposition.object}`);
    for (const match of output.matches || [])
        card('Verified match · ' + Number(match.similarity).toFixed(3), match.claim.text, [], match.claim.source + ' · ' + match.claim.original_sentence);
    for (const pair of output.comparisons || [])
        card(pair.decision.verdict, pair.claim.text, [pair.claim], 'Compared with: ' + pair.verified_claim.text);
    if (data.record)
        card('Unverified record', data.record.text, [data.record]);
    if (data.decision)
        card(data.decision.verdict, data.verified_claim?.text || 'Comparison completed', [], 'Retrieved source: ' + (data.verified_claim?.source || ''));
    if (output.text)
        card('Output', output.text, events[selected]?.spans || []);
    if (output.verdict)
        card('Decision', output.verdict, events[selected]?.spans || []);
    if (data.kind && data.text)
        card(data.kind, data.text, events[selected]?.spans || []);
    if (!data.output && output.markdown)
        card('Rendered Markdown', output.markdown);
    if (!$('cards').children.length && Object.keys(output).length) {
        const pairs = Object.entries(output).filter(([, v]) => typeof v === 'boolean' || typeof v === 'number' || typeof v === 'string');
        for (const [key, value] of pairs.slice(0, 6))
            card(key.replaceAll('_', ' '), value, events[selected]?.spans || []);
    }
}
function chart(data) {
    $('chart').replaceChildren();
    if (!data.similarity)
        return;
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 360 150');
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Prose gap similarity and depth scores; list contents excluded');
    const all = [...data.similarity, ...data.depths, 0, 1];
    const low = Math.min(...all), high = Math.max(...all), n = data.similarity.length;
    const x = i => 25 + i * 315 / Math.max(n - 1, 1), y = v => 125 - (v - low) / (high - low) * 105;
    for (const value of [low, high]) {
        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', '0');
        label.setAttribute('y', y(value) + 3);
        label.setAttribute('font-size', '9');
        label.textContent = value.toFixed(1);
        svg.append(label);
    }
    for (const [values, color] of [[data.similarity, '#187962'], [data.depths, '#c2753b']]) {
        const line = document.createElementNS(ns, 'polyline');
        line.setAttribute('points', values.map((v, i) => `${x(i)},${y(v)}`).join(' '));
        line.setAttribute('fill', 'none');
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', '2');
        svg.append(line);
    }
    for (const boundary of data.boundaries || []) {
        let i = data.positions.reduce((best, p, index) => Math.abs(p - boundary) < Math.abs(data.positions[best] - boundary) ? index : best, 0);
        const line = document.createElementNS(ns, 'line');
        for (const [key, v] of Object.entries({ x1: x(i), x2: x(i), y1: 15, y2: 130, stroke: '#a3b0a5', 'stroke-dasharray': '3 3' }))
            line.setAttribute(key, v);
        svg.append(line);
    }
    $('chart').append(svg);
    const legend = element('p', undefined, 'chart-legend');
    legend.append(element('span', '— Similarity'), element('span', '— Depth'));
    $('chart').append(legend, element('p', (data.boundaries || []).length + ' selected boundaries · lists excluded. Click a gap to locate it.'));
    data.positions.forEach((position, i) => {
        const dot = document.createElementNS(ns, 'circle');
        dot.setAttribute('cx', x(i));
        dot.setAttribute('cy', y(data.similarity[i]));
        dot.setAttribute('r', '4');
        dot.setAttribute('fill', '#187962');
        dot.setAttribute('tabindex', '0');
        dot.setAttribute('role', 'button');
        dot.setAttribute('aria-label', 'Gap at character ' + position);
        const activate = () => highlight([{ start: position, end: Math.min(Array.from(source).length, position + 60) }]);
        dot.onclick = activate;
        dot.onkeydown = e => {
            if (e.key === 'Enter' || e.key === ' ')
                activate();
        };
        svg.append(dot);
    });
}
function selectEvent(index) {
    if (index < 0 || index >= events.length)
        return;
    selected = index;
    const event = events[index], description = descriptions[event.stage] || [event.stage.replaceAll('_', ' '), 'Inspect the result and its original source.'];
    summary(description[0], event.status === 'failed' ? (event.data.error || 'This stage failed.') : description[1]);
    highlight(event.document_id === currentRun?.document_id ? event.spans : []);
    $('details').textContent = JSON.stringify(event.data, null, 2);
    chart(event.data);
    showCards(event.data);
    for (const phase of workflowProgress(task, events)) {
        const node = $('step-' + phase.id);
        node.classList.toggle('selected', phase.indices.includes(index));
        if (phase.indices.includes(index)) node.querySelector('select').value = String(index);
    }
}
function buildSteps() {
    $('timeline').replaceChildren();
    for (const [id, title] of workflowSteps(task)) {
        const item = element('li'), details = element('details', undefined, 'workflow-step');
        details.id = 'step-' + id;
        const heading = element('summary');
        heading.append(element('span', title), element('small', 'Waiting', 'phase-state'));
        const label = element('label', 'Inspect evidence');
        const select = element('select');
        select.setAttribute('aria-label', title + ' evidence');
        select.disabled = true;
        select.onchange = () => {
            $('follow').checked = false;
            selectEvent(Number(select.value));
        };
        label.append(select);
        const inspect = element('button', 'Show selected evidence');
        inspect.disabled = true;
        inspect.onclick = () => {
            $('follow').checked = false;
            selectEvent(Number(select.value));
        };
        details.append(heading, label, inspect);
        item.append(details);
        $('timeline').append(item);
    }
}
function updateSteps() {
    const phases = workflowProgress(task, events);
    for (const phase of phases) {
        const node = $('step-' + phase.id), select = node.querySelector('select');
        node.querySelector('.phase-state').textContent = phase.state;
        const previous = select.value;
        select.replaceChildren();
        for (const index of phase.indices) {
            const event = events[index];
            const label = (descriptions[event.stage]?.[0] || event.stage)
                + (Number.isInteger(event.data.sentence_index) ? ' · sentence ' + (event.data.sentence_index + 1) : '')
                + ' · ' + event.status;
            const option = element('option', label);
            option.value = String(index);
            select.append(option);
        }
        select.disabled = !phase.indices.length;
        node.querySelector('button').disabled = !phase.indices.length;
        select.value = !$('follow').checked && phase.indices.includes(Number(previous)) && previous !== ''
            ? previous : String(phase.indices.at(-1));
    }
    $('event-count').textContent = phases.filter(phase => phase.indices.length).length + ' / ' + phases.length + ' stages';
}
function appendEvent(event) {
    if (events.some(existing => existing.sequence === event.sequence)) return;
    events.push(event);
    updateSteps();
    if ($('follow').checked) selectEvent(events.length - 1);
}
function moveStep(direction) {
    const phases = workflowProgress(task, events).filter(phase => phase.indices.length);
    const current = phases.findIndex(phase => phase.indices.includes(selected));
    const next = phases[current < 0 ? (direction > 0 ? 0 : phases.length - 1) : current + direction];
    if (!next) return;
    $('follow').checked = false;
    $('step-' + next.id).open = true;
    selectEvent(next.indices.at(-1));
}
function requestBody() {
    return { task, source, source_name: sourceName, parent_run_id: task === 'claim_store' ? lastClaimify : null, method: $('method').value, detect_headings: $('detect').checked, generate_headings: $('headings').checked, generation_model: $('generation-model').value, decision_model: $('decision-model').value, embedding_model: $('embedding-model').value || null, local_model_path: $('local-path').value || null, local_files_only: $('local-only').checked, max_concurrency: Number($('concurrency').value), question: $('question').value || null, entity_model: $('entity-model').value, labels: $('labels').value.split(',').map(s => s.trim()).filter(Boolean), threshold: Number($('threshold').value), store_action: $('store-action').value, top_k: Number($('top-k').value) };
}
async function run() {
    if (busy) return;
    const body = requestBody();
    resetView();
    setBusy(true);
    $('follow').checked = true;
    status('Starting live task…');
    try {
        currentRun = await api('/api/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        history.replaceState(null, '', '?run=' + currentRun.id);
        source = currentRun.source;
        sourceName = currentRun.source_name;
        $('source-name').textContent = sourceName;
        $('char-count').textContent = Array.from(source).length.toLocaleString() + ' chars';
        highlight([]);
        followRun();
    }
    catch (error) {
        status(error.message, true);
        setBusy(false);
    }
}
function followRun() {
    const runId = currentRun.id;
    if (stream) stream.close();
    const connection = new EventSource('/api/runs/' + runId + '/events');
    stream = connection;
    $('cancel').disabled = stopping || !['running', 'cancelling'].includes(currentRun.status);
    connection.onmessage = async message => {
        if (stream !== connection || currentRun?.id !== runId) return;
        const event = JSON.parse(message.data);
        const terminal = event.stage === 'run' && ['completed', 'failed', 'cancelled'].includes(event.status);
        if (terminal) connection.close();
        try {
            appendEvent(event);
            status(stopping || event.status === 'cancelling' ? 'Stopping task · waiting for active work' : descriptions[event.stage]?.[0] || event.stage, event.status === 'failed');
        } catch (error) {
            status('Could not display a processing step: ' + error.message, true);
        } finally {
            if (terminal) await finish(runId);
        }
    };
    connection.onerror = async () => {
        if (stream !== connection || currentRun?.id !== runId) return;
        // A missed terminal event must not strand the UI in a running state.
        try {
            const latest = await api('/api/runs/' + runId);
            if (stream !== connection || currentRun?.id !== runId) return;
            if (['completed', 'failed', 'cancelled'].includes(latest.status)) {
                connection.close();
                await finish(runId);
            } else status('Stream disconnected · reconnecting to the same run', true);
        } catch (error) {
            if (stream !== connection || currentRun?.id !== runId) return;
            connection.close();
            setBusy(false);
            status('Could not reconnect to this run: ' + error.message, true);
        }
    };
}
async function finish(runId = currentRun?.id) {
    try {
        const completed = await api('/api/runs/' + runId);
        if (currentRun?.id !== runId) return;
        currentRun = completed;
        result = currentRun.result;
        const failedSentences = result?.sentences?.filter(sentence => sentence.status === 'validation_failed').length || 0;
        status(currentRun.status === 'failed' ? currentRun.error : currentRun.status === 'cancelled' ? 'Task stopped'
            : failedSentences ? 'Run complete · ' + failedSentences + ' sentences need review' : 'Run complete · select any step to review', currentRun.status === 'failed');
        if (result) {
            $('download').disabled = false;
            showFinalResult();
            if ($('follow').checked) {
                summary('Processing complete', 'Select a result to highlight its original evidence, or inspect any step in the timeline.');
                showCards(result);
            }
            if (task === 'claimify') lastClaimify = currentRun.id;
            handoffs();
        }
        if (task === 'claim_store') await refreshClaims();
    } catch (error) {
        if (currentRun?.id === runId) status(error.message, true);
    } finally {
        if (currentRun?.id === runId) {
            stopping = false;
            $('cancel').textContent = 'Stop task';
            setBusy(false);
        }
    }
}
function handoffs() {
    $('handoffs').replaceChildren();
    if (task === 'segmentation' && result?.markdown) {
        for (const target of ['claimify', 'openie']) {
            const button = element('button', 'Use in ' + target + ' →');
            button.onclick = () => {
                const markdown = result.markdown;
                selectTask(target);
                setSource(markdown, sourceName.replace(/\.md$/, '') + '.segmented.md');
            };
            $('handoffs').append(button);
        }
    }
    if (task === 'claimify') {
        const button = element('button', 'Send to claim store →');
        button.onclick = () => selectTask('claim_store');
        $('handoffs').append(button);
    }
}
async function refreshClaims() {
    try {
        const data = await api('/api/claims');
        $('store-records').replaceChildren();
        for (const name of ['unverified', 'verified']) {
            const column = element('div');
            column.append(element('h3', name + ' · ' + data[name].length), element('pre', data[name].map(record => record.text + '\n' + record.id + '.json').join('\n\n') || 'No records yet.'));
            $('store-records').append(column);
        }
    }
    catch (error) {
        status(error.message, true);
    }
}
for (const [index, [value, title]] of tasks.entries()) {
    const button = element('button');
    button.dataset.task = value;
    button.append(element('small', String(index + 1).padStart(2, '0')), document.createTextNode(title));
    button.onclick = () => selectTask(value);
    $('tabs').append(button);
}
$('run').onclick = run;
$('cancel').onclick = async () => {
    if (!currentRun || stopping) return;
    stopping = true;
    $('cancel').disabled = true;
    $('cancel').textContent = 'Stopping…';
    status('Stopping task · waiting for active work to finish');
    try {
        await api('/api/runs/' + currentRun.id + '/cancel', { method: 'POST' });
    } catch (error) {
        stopping = false;
        $('cancel').disabled = !busy;
        $('cancel').textContent = 'Stop task';
        status(error.message, true);
    }
};
$('prev').onclick = () => moveStep(-1);
$('next').onclick = () => moveStep(1);
$('follow').onchange = () => {
    if ($('follow').checked)
        selectEvent(events.length - 1);
};
$('edit').onclick = () => {
    $('editor-panel').hidden = !$('editor-panel').hidden;
};
$('apply').onclick = () => {
    setSource($('editor').value, 'pasted-article.md');
    $('editor-panel').hidden = true;
};
$('file').onchange = async (event) => {
    const file = event.target.files[0];
    if (file)
        setSource(await file.text(), file.name);
};
$('refresh-claims').onclick = refreshClaims;
$('download').onclick = () => {
    const blob = new Blob([JSON.stringify({ ...currentRun, events }, null, 2)], { type: 'application/json' }), url = URL.createObjectURL(blob), a = element('a');
    a.href = url;
    a.download = task + '-' + currentRun.id.slice(0, 8) + '.json';
    a.click();
    URL.revokeObjectURL(url);
};
function markdownPreview(markdown) {
    const frame = element('iframe', undefined, 'markdown-preview');
    frame.title = 'Rendered Markdown preview';
    frame.setAttribute('sandbox', '');
    api('/api/preview', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({task: 'segmentation', source: markdown})})
        .then(data => { frame.srcdoc = data.html; })
        .catch(error => { frame.replaceWith(element('pre', 'Preview unavailable: ' + error.message + '\n\n' + markdown)); });
    return frame;
}
function showFinalResult() {
    $('final-panel').hidden = false;
    $('view-final').hidden = false;
    $('final-title').textContent = task === 'segmentation' ? 'Final segmented Markdown' : 'Final result';
    const counts = {};
    for (const sentence of result.sentences || []) counts[sentence.status] = (counts[sentence.status] || 0) + 1;
    const outcome = Array.isArray(result.claims)
        ? ' · ' + result.claims.length + ' claims · ' + Object.entries(counts).map(([name, count]) => count + ' ' + name.replaceAll('_', ' ')).join(' · ')
        : '';
    $('final-summary').textContent = tasks.find(item => item[0] === task)[1] + ' · ' + sourceName + outcome;
    $('final-json').textContent = JSON.stringify(result, null, 2);
    const output = $('final-output');
    output.replaceChildren();
    if (typeof result.markdown === 'string') {
        const markdown = result.markdown;
        const controls = element('div', undefined, 'markdown-controls');
        const rendered = element('button', 'Rendered preview');
        const raw = element('button', 'Markdown source');
        const download = element('button', 'Download .md');
        const preview = markdownPreview(markdown);
        const code = element('pre', markdown);
        code.hidden = true;
        rendered.setAttribute('aria-pressed', 'true');
        raw.setAttribute('aria-pressed', 'false');
        function chooseView(showSource) {
            preview.hidden = showSource;
            code.hidden = !showSource;
            rendered.setAttribute('aria-pressed', String(!showSource));
            raw.setAttribute('aria-pressed', String(showSource));
        }
        rendered.onclick = () => chooseView(false);
        raw.onclick = () => chooseView(true);
        download.onclick = () => {
            const url = URL.createObjectURL(new Blob([markdown], {type: 'text/markdown;charset=utf-8'}));
            const link = element('a');
            link.href = url;
            link.download = sourceName.split(/[\\/]/).pop().replace(/\.md$/i, '') + '.segmented.md';
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        };
        controls.append(rendered, raw, download);
        output.append(controls, preview, code);
        return;
    }
    const collections = Object.entries(result).filter(([, value]) => Array.isArray(value))
        .sort(([a], [b]) => (a === 'claims' ? -1 : 0) - (b === 'claims' ? -1 : 0));
    for (const [name, rows] of collections) {
        output.append(element('h3', name.replaceAll('_', ' ') + ' · ' + rows.length));
        if (!rows.length) {
            output.append(element('p', 'No ' + name.replaceAll('_', ' ') + ' returned.'));
            continue;
        }
        const columns = name === 'sentences'
            ? ['index', 'status', 'text', 'selected_text', 'clarified_text']
            : [...new Set(rows.flatMap(row => row && typeof row === 'object' ? Object.keys(row) : ['value']))];
        const table = element('table'), head = element('thead'), heading = element('tr'), body = element('tbody');
        for (const key of columns) heading.append(element('th', key.replaceAll('_', ' ')));
        head.append(heading);
        for (const row of rows) {
            const tr = element('tr'), record = row && typeof row === 'object' ? row : {value: row};
            for (const key of columns) {
                const value = record[key];
                tr.append(element('td', value == null ? '' : typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)));
            }
            body.append(tr);
        }
        table.append(head, body);
        const wrapper = element('div', undefined, 'result-table');
        wrapper.append(table);
        output.append(wrapper);
    }
    if (!collections.length) output.append(element('pre', JSON.stringify(result, null, 2)));
}
$('save-result').onclick = async () => {
    const runId = currentRun.id;
    $('save-result').disabled = true;
    $('save-status').textContent = 'Saving…';
    try {
        const saved = await api('/api/runs/' + runId + '/save', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({directory: $('save-directory').value.trim() || null})});
        if (currentRun?.id === runId) $('save-status').textContent = 'Saved: ' + saved.files.join(' · ');
    } catch (error) {
        if (currentRun?.id === runId) {
            $('save-status').textContent = error.message;
            $('save-result').disabled = false;
        }
    } finally {
        if (currentRun?.id === runId) $('save-result').disabled = false;
    }
};
async function loadDocument() {
    if (!$('example').value) return;
    setBusy(true);
    try {
        const document = await api('/api/documents/content?path=' + encodeURIComponent($('example').value));
        setSource(document.source, document.name);
    } catch (error) { status(error.message, true); }
    finally { setBusy(false); }
}
async function refreshDocuments() {
    setBusy(true);
    try {
        const previous = $('example').value;
        const documents = await api('/api/documents');
        $('example').replaceChildren();
        for (const document of documents) {
            const option = element('option', document.name);
            option.value = document.name;
            $('example').append(option);
        }
        if (documents.some(document => document.name === previous)) $('example').value = previous;
        if (documents.length) await loadDocument();
        else status('No Markdown files in the data directory. Open a file or paste a source.');
    } catch (error) { status(error.message, true); }
    finally { setBusy(false); }
}
$('example').onchange = loadDocument;
$('refresh-documents').onclick = refreshDocuments;
(async () => {
    selectTask('segmentation');
    await refreshDocuments();
    setBusy(true);
    try {
        const capabilities = await api('/api/capabilities');
        $('save-directory').value = capabilities.result_root;
        const runId = new URLSearchParams(location.search).get('run');
        if (runId) {
            const existing = await api('/api/runs/' + encodeURIComponent(runId));
            selectTask(existing.task);
            setSource(existing.source, existing.source_name);
            currentRun = existing;
            setBusy(true);
            $('follow').checked = false;
            followRun();
        }
        $('readiness').textContent = Object.entries(capabilities.providers).map(([key, ready]) => key.replace('_API_KEY', '') + ': ' + (ready ? 'ready' : 'not configured')).join(' · ');
        $('store-description').textContent = 'Files: ' + capabilities.store_root + '. To approve a reviewed record, manually move its JSON from unverified/ to verified/, then refresh. The application never approves claims.';
    } catch (error) { status(error.message, true); }
    finally { if (!stream) setBusy(false); }
})();

function setSourceView(view) {
    sourceView = view;
    const preview = view === 'preview';
    $('source').hidden = preview;
    $('source-preview').hidden = !preview;
    $('source-raw-button').setAttribute('aria-pressed', String(!preview));
    $('source-preview-button').setAttribute('aria-pressed', String(preview));
    if (preview && !$('source-preview').children.length)
        $('source-preview').append(markdownPreview(source));
}
$('source-raw-button').onclick = () => setSourceView('raw');
$('source-preview-button').onclick = () => setSourceView('preview');

$('view-final').onclick = () => $('final-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
