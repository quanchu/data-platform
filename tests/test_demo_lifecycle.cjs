const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../src/data_platform/demo/static/app.js'), 'utf8');
const connections = [], calls = [], controls = {};
let resolveResult;
const context = vm.createContext({
    currentRun: {id:'first',status:'running'}, stream:null, stopping:false, result:null, task:'segmentation',
    EventSource: class { constructor() { connections.push(this); } close() { this.closed = true; } },
    $: id => controls[id] ||= {checked:false},
    api: async path => { calls.push(path); return new Promise(resolve => { resolveResult = resolve; }); },
    appendEvent: () => { throw Error('render failure'); }, status: () => {}, descriptions:{},
    setBusy: value => calls.push(['busy', value]), showFinalResult: () => {}, showCards: () => {},
    summary: () => {}, handoffs: () => {}, refreshClaims: async () => {}
});
vm.runInContext(source.slice(source.indexOf('function followRun()'), source.indexOf('function handoffs()')), context);
(async () => {
    context.followRun();
    const old = connections[0];
    context.currentRun = {id:'second',status:'running'};
    context.followRun();
    assert.equal(old.closed, true);
    await old.onmessage({data: JSON.stringify({stage:'run',status:'completed'})});
    assert.equal(calls.length, 0, 'Old stream must not fetch or update the new run');
    const finish = connections[1].onmessage({data: JSON.stringify({stage:'run',status:'completed'})});
    assert.equal(connections[1].closed, true);
    assert.equal(calls[0], '/api/runs/second', 'Rendering failure must still retrieve terminal result');
    resolveResult({id:'second',status:'completed',result:{markdown:'# Result'}});
    await finish;
    assert.equal(context.result.markdown, '# Result');
    const pending = context.finish('second');
    context.currentRun = {id:'third',status:'running'};
    const priorCalls = calls.length;
    resolveResult({id:'second',status:'completed',result:{markdown:'stale'}});
    await pending;
    assert.equal(context.currentRun.id, 'third');
    assert.equal(calls.length, priorCalls, 'Old finish must not enable controls for new run');
    console.log('Repeated-run stream isolation and terminal recovery checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
