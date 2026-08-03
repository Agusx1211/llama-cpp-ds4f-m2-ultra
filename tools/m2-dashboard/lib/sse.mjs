export class SseParser {
    constructor(onMessage) {
        if (typeof onMessage !== "function") {
            throw new TypeError("onMessage must be a function");
        }
        this.onMessage = onMessage;
        this.buffer = "";
        this.data = [];
        this.eventId = "";
        this.eventType = "message";
    }

    push(chunk) {
        this.buffer += chunk;
        while (true) {
            const newline = this.buffer.search(/[\r\n]/);
            if (newline < 0 || (this.buffer[newline] === "\r" && newline === this.buffer.length - 1)) {
                break;
            }
            const width = this.buffer[newline] === "\r" && this.buffer[newline + 1] === "\n" ? 2 : 1;
            const line = this.buffer.slice(0, newline);
            this.buffer = this.buffer.slice(newline + width);
            this.consumeLine(line);
        }
    }

    finish() {
        if (this.buffer.length > 0) {
            const line = this.buffer.endsWith("\r") ? this.buffer.slice(0, -1) : this.buffer;
            this.consumeLine(line);
            this.buffer = "";
        }
        this.dispatch();
    }

    consumeLine(line) {
        if (line === "") {
            this.dispatch();
            return;
        }
        if (line.startsWith(":")) {
            return;
        }

        const separator = line.indexOf(":");
        const field = separator < 0 ? line : line.slice(0, separator);
        let value = separator < 0 ? "" : line.slice(separator + 1);
        if (value.startsWith(" ")) {
            value = value.slice(1);
        }

        switch (field) {
            case "data":
                this.data.push(value);
                break;
            case "event":
                this.eventType = value || "message";
                break;
            case "id":
                if (!value.includes("\0")) {
                    this.eventId = value;
                }
                break;
            default:
                break;
        }
    }

    dispatch() {
        if (this.data.length === 0) {
            this.eventType = "message";
            return;
        }
        this.onMessage({
            data: this.data.join("\n"),
            lastEventId: this.eventId,
            type: this.eventType,
        });
        this.data = [];
        this.eventType = "message";
    }
}

export function createFetchSseTransport(url, fetchImpl = globalThis.fetch) {
    if (typeof fetchImpl !== "function") {
        throw new Error("fetch is unavailable");
    }

    return ({ lastEventId, onEvent, onDisconnect }) => {
        const controller = new AbortController();
        let closed = false;

        void (async () => {
            try {
                const headers = { Accept: "text/event-stream" };
                if (lastEventId !== null && lastEventId !== undefined && String(lastEventId) !== "") {
                    headers["Last-Event-ID"] = String(lastEventId);
                }
                const response = await fetchImpl(url, {
                    method: "GET",
                    headers,
                    cache: "no-store",
                    credentials: "same-origin",
                    signal: controller.signal,
                });
                if (!response.ok || response.body === null) {
                    throw new Error(`SSE request failed: ${response.status} ${response.statusText}`);
                }

                const parser = new SseParser(onEvent);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                while (!closed) {
                    const { value, done } = await reader.read();
                    if (done) {
                        parser.push(decoder.decode());
                        parser.finish();
                        break;
                    }
                    parser.push(decoder.decode(value, { stream: true }));
                }
                if (!closed) {
                    onDisconnect(new Error("SSE stream ended"));
                }
            } catch (error) {
                if (!closed && error?.name !== "AbortError") {
                    onDisconnect(error);
                }
            }
        })();

        return () => {
            closed = true;
            controller.abort();
        };
    };
}
