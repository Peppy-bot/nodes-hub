const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const script = readFileSync(path.join(__dirname, "..", "browser_logs.js"), "utf8");

function viewer(fetchResult = () => Promise.resolve()) {
    const requests = [];
    const calls = [];
    const events = new Map();
    const promises = [];
    const console = Object.fromEntries(["warn", "error", "info", "log"].map(level => [
        level, function (...args) {
            calls.push({ level, args, receiver: this });
            return "original result";
        },
    ]));
    const context = vm.createContext({
        Error,
        console,
        window: {
            location: { origin: "http://100.92.223.49:8210", pathname: "/", search: "?token=private" },
            addEventListener: (type, callback) => events.set(type, callback),
        },
        fetch(url, options) {
            requests.push({ url, options, entry: JSON.parse(options.body) });
            const result = fetchResult();
            promises.push(result);
            return result;
        },
    });
    vm.runInContext(script, context);
    return {
        context, console, requests, calls, events,
        async settled() {
            await Promise.allSettled(promises);
            await new Promise(resolve => setImmediate(resolve));
        },
    };
}

test("page-open report is same-origin, excludes query data, and does not alter info/log", async () => {
    const v = viewer();
    assert.equal(v.requests.length, 1);
    const { url, options, entry } = v.requests[0];
    assert.equal(url, "/browser-logs");
    assert.equal(options.method, "POST");
    assert.equal(options.mode, "same-origin");
    assert.equal(options.credentials, "omit");
    assert.equal(options.keepalive, true);
    assert.equal(options.headers["Content-Type"], "application/json");
    assert.deepEqual(entry, {
        level: "info", message: "Isaac Sim viewer page loaded", page: "http://100.92.223.49:8210/",
    });
    v.console.info("info");
    v.console.log("log");
    assert.equal(v.requests.length, 1);
    assert.equal(v.calls.length, 2);
    await v.settled();
});

test("forwards the NVIDIA screenshot callback and preserves original console behavior", async () => {
    const v = viewer();
    const message = 'Session start failed with error: "Error occurred during sign-in request to signaling server."';
    assert.equal(v.console.error("Stream start error:", message), "original result");
    v.console.warn({ type: "warning", detail: "1 of 6 connection attempts have failed. Retrying..." });
    assert.equal(v.calls[0].receiver, v.console);
    assert.deepEqual(v.calls[0].args, ["Stream start error:", message]);
    assert.equal(v.requests[1].entry.level, "error");
    assert.equal(v.requests[1].entry.message, "Stream start error: " + message);
    assert.equal(v.requests[2].entry.level, "warn");
    assert.equal(JSON.parse(v.requests[2].entry.message).type, "warning");
    await v.settled();
});

test("captures uncaught errors and unhandled rejections without swallowing them", async () => {
    const v = viewer();
    const error = new Error("uncaught failure");
    v.events.get("error")({ error, preventDefault() { assert.fail("must not suppress errors"); } });
    v.events.get("error")({ message: "Script error." });
    v.events.get("unhandledrejection")({ reason: new Error("rejected failure") });
    assert.match(v.requests[1].entry.message, /Uncaught browser error: Error: uncaught failure/);
    assert.match(v.requests[1].entry.message, /browser_logs.test.js/);
    assert.equal(v.requests[2].entry.message, "Uncaught browser error: Script error.");
    assert.match(v.requests[3].entry.message, /Unhandled browser rejection: Error: rejected failure/);
    assert.equal(v.calls.length, 0);
    await v.settled();
});

test("serialization handles cycles, bigint, empty values and throwing serializers", async () => {
    const v = viewer();
    const circular = { number: 5n };
    circular.self = circular;
    v.console.error(circular, undefined, null);
    v.console.warn({ toJSON() { throw new Error("cannot serialize"); } });
    v.console.warn();
    assert.equal(v.requests[1].entry.message, '{"number":"5","self":"[Circular]"} undefined null');
    assert.equal(v.requests[2].entry.message, "[Unserializable browser log value]");
    assert.equal(v.requests[3].entry.message, "(empty)");
    await v.settled();
});

test("preserves repeated acyclic references while marking actual cycles", async () => {
    const v = viewer();
    const shared = { code: 503 };
    v.console.error({ first: shared, second: shared });
    assert.deepEqual(JSON.parse(v.requests[1].entry.message), { first: { code: 503 }, second: { code: 503 } });
    await v.settled();
});

test("bounds object traversal rather than visiting every diagnostic field", async () => {
    const v = viewer();
    let reads = 0;
    const large = Array.from({ length: 10_000 }, () => ({ get value() { reads += 1; return "x"; } }));
    v.console.error("Connection failed:", large);
    assert.ok(reads > 0 && reads <= 128);
    assert.equal(v.requests[1].entry.message, "Connection failed: [Browser log value exceeds serialization limit]");
    assert.equal(v.calls[0].args[1], large);
    await v.settled();
});

test("truncates reports so even escaped characters stay within the server body limit", async () => {
    const v = viewer();
    v.console.error("\0".repeat(30_000));
    assert.equal(v.requests[1].entry.message.length, 2048);
    assert.ok(Buffer.byteLength(v.requests[1].options.body) <= 16_384);
    assert.equal(v.calls[0].args[0].length, 30_000);
    await v.settled();
});

test("does not serialize arguments that cannot fit in the transmitted message", async () => {
    const v = viewer();
    let reads = 0;
    const unused = { get detail() { reads += 1; return "unused"; } };
    v.console.error("x".repeat(2048), unused);
    v.console.error("x".repeat(2047), unused);
    assert.equal(reads, 0);
    assert.equal(v.requests[1].entry.message, "x".repeat(2048));
    assert.equal(v.requests[2].entry.message, "x".repeat(2047) + " ");
    await v.settled();
});

test("rejected logging requests never recurse and release their pending slots", async () => {
    const v = viewer(() => Promise.reject(new Error("log endpoint unavailable")));
    for (let i = 0; i < 20; i += 1) {
        v.console.error("stream failure", i);
        await v.settled();
    }
    assert.equal(v.requests.length, 21);
    assert.equal(v.calls.length, 20);
});

test("synchronous reporting failures do not change console behavior", () => {
    const v = viewer(() => { throw new Error("fetch unavailable"); });
    for (let i = 0; i < 20; i += 1) {
        assert.equal(v.console.error("stream failure"), "original result");
    }
    assert.equal(v.requests.length, 21);
    assert.equal(v.calls.length, 20);
});

test("an unresponsive endpoint bounds pending requests without suppressing console output", async () => {
    const resolvers = [];
    const v = viewer(() => new Promise(resolve => resolvers.push(resolve)));
    for (let i = 0; i < 100; i += 1) v.console.error("stream failure");
    assert.equal(v.requests.length, 8);
    assert.equal(v.calls.length, 100);
    for (const resolve of resolvers) resolve();
    await v.settled();
    v.console.error("new failure");
    assert.equal(v.requests.length, 9);
    resolvers.at(-1)();
    await v.settled();
});
