(() => {
    const MAX_MESSAGE_CHARS = 2048;
    const MAX_PENDING = 8;
    const SERIALIZATION_LIMIT = Symbol("serialization limit");
    const page = (window.location.origin + window.location.pathname).slice(0, 256);
    let pending = 0;

    function describe(value) {
        try {
            if (typeof value === "string") return value;
            if (value instanceof Error) return value.stack || value.message;
            const ancestors = [];
            let visited = 0;
            return JSON.stringify(value, function (key, item) {
                visited += 1;
                if (visited > 128 || key.length > MAX_MESSAGE_CHARS) throw SERIALIZATION_LIMIT;
                if (typeof item === "string") return item.slice(0, MAX_MESSAGE_CHARS);
                if (typeof item === "bigint") return String(item).slice(0, MAX_MESSAGE_CHARS);
                if (item && typeof item === "object") {
                    while (ancestors.length && ancestors.at(-1) !== this) ancestors.pop();
                    if (ancestors.includes(item)) return "[Circular]";
                    ancestors.push(item);
                }
                return item;
            }) ?? String(value);
        } catch (error) {
            return error === SERIALIZATION_LIMIT
                ? "[Browser log value exceeds serialization limit]"
                : "[Unserializable browser log value]";
        }
    }

    function report(level, args) {
        if (pending >= MAX_PENDING) return;
        pending += 1;
        try {
            let message = "";
            for (const [index, value] of args.slice(0, 20).entries()) {
                if (index > 0 && message.length < MAX_MESSAGE_CHARS) message += " ";
                const remaining = MAX_MESSAGE_CHARS - message.length;
                if (remaining === 0) break;
                message += describe(value).slice(0, remaining);
            }
            void fetch("/browser-logs", {
                method: "POST",
                mode: "same-origin",
                credentials: "omit",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ level, message: message || "(empty)", page }),
                keepalive: true,
            }).catch(() => {}).finally(() => { pending -= 1; });
        } catch {
            // Reporting must not break the viewer or recursively log its own failure.
            pending -= 1;
        }
    }

    for (const level of ["warn", "error"]) {
        const original = console[level].bind(console);
        console[level] = (...args) => {
            const result = original(...args);
            report(level, args);
            return result;
        };
    }

    window.addEventListener("error", (event) => {
        report("error", [
            "Uncaught browser error:",
            event.error || event.message || "Resource failed to load",
        ]);
    }, true);
    window.addEventListener("unhandledrejection", (event) => {
        report("error", ["Unhandled browser rejection:", event.reason]);
    });
    report("info", ["Isaac Sim viewer page loaded"]);
})();
