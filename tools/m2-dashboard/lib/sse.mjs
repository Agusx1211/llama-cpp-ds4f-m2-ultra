export const DEFAULT_MAX_SSE_BUFFER_BYTES = 2 * 1024 * 1024;
export const DEFAULT_MAX_SSE_EVENT_BYTES = 2 * 1024 * 1024;

const textEncoder = new TextEncoder();

function byteLength(value) {
    return textEncoder.encode(value).byteLength;
}

function boundedByteLength(value, limit) {
    if (limit < 0 || value.length > limit) {
        return limit + 1;
    }
    return byteLength(value);
}

function positiveByteLimit(value, name) {
    if (!Number.isSafeInteger(value) || value < 1) {
        throw new RangeError(`${name} must be a positive integer`);
    }
    return value;
}

export class SseParser {
    constructor(onMessage, {
        maxBufferBytes = DEFAULT_MAX_SSE_BUFFER_BYTES,
        maxEventBytes = DEFAULT_MAX_SSE_EVENT_BYTES,
    } = {}) {
        if (typeof onMessage !== "function") {
            throw new TypeError("onMessage must be a function");
        }
        this.onMessage = onMessage;
        this.maxBufferBytes = positiveByteLimit(maxBufferBytes, "maxBufferBytes");
        this.maxEventBytes = positiveByteLimit(maxEventBytes, "maxEventBytes");
        this.buffer = "";
        this.bufferBytes = 0;
        this.data = [];
        this.eventId = "";
        this.eventType = "message";
        this.eventBytes = 0;
    }

    push(chunk) {
        if (typeof chunk !== "string") {
            throw new TypeError("SSE chunk must be a string");
        }
        const priorBufferBytes = this.bufferBytes;
        this.buffer += chunk;
        let consumedLine = false;
        while (true) {
            const newline = this.buffer.search(/[\r\n]/);
            if (newline < 0 || (this.buffer[newline] === "\r" && newline === this.buffer.length - 1)) {
                break;
            }
            const width = this.buffer[newline] === "\r" && this.buffer[newline + 1] === "\n" ? 2 : 1;
            const line = this.buffer.slice(0, newline);
            this.buffer = this.buffer.slice(newline + width);
            consumedLine = true;
            this.addEventText(line, width);
            this.consumeLine(line);
        }
        if (consumedLine) {
            this.bufferBytes = boundedByteLength(this.buffer, this.maxBufferBytes);
        } else {
            this.bufferBytes = priorBufferBytes + boundedByteLength(
                chunk,
                this.maxBufferBytes - priorBufferBytes,
            );
        }
        if (this.bufferBytes > this.maxBufferBytes) {
            throw new RangeError(`SSE line buffer exceeds ${this.maxBufferBytes} bytes`);
        }
    }

    finish() {
        if (this.buffer.length > 0) {
            const line = this.buffer.endsWith("\r") ? this.buffer.slice(0, -1) : this.buffer;
            this.addEventText(this.buffer);
            this.consumeLine(line);
            this.buffer = "";
            this.bufferBytes = 0;
        }
        this.dispatch();
    }

    addEventText(value, structuralBytes = 0) {
        const remaining = this.maxEventBytes - this.eventBytes - structuralBytes;
        const valueBytes = boundedByteLength(value, remaining);
        if (valueBytes > remaining) {
            throw new RangeError(`SSE event exceeds ${this.maxEventBytes} bytes`);
        }
        this.eventBytes += valueBytes + structuralBytes;
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
            this.eventBytes = 0;
            return;
        }
        this.onMessage({
            data: this.data.join("\n"),
            lastEventId: this.eventId,
            type: this.eventType,
        });
        this.data = [];
        this.eventType = "message";
        this.eventBytes = 0;
    }
}

export function createFetchSseTransport(url, fetchImpl = globalThis.fetch, parserOptions = {}) {
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
                    const error = new Error(`SSE request failed: ${response.status} ${response.statusText}`);
                    if (response.status === 409) {
                        error.resnapshotRequired = true;
                    }
                    throw error;
                }

                const parser = new SseParser(onEvent, parserOptions);
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
