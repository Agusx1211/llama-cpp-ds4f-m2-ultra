const REQUEST_ID = /^[1-9]\d*:[1-9]\d*$/u;

export function cancelConfirmation(requestId) {
    if (typeof requestId !== "string" || requestId.length > 41 || !REQUEST_ID.test(requestId)) {
        throw new TypeError("request ID must be canonical id:epoch");
    }
    return `Cancel live request ${requestId}? This stops generation and cannot be undone.`;
}

export async function confirmAndCancel({ requestId, cancelRequest, confirmImpl = globalThis.confirm }) {
    if (typeof cancelRequest !== "function") {
        throw new TypeError("cancelRequest must be a function");
    }
    if (typeof confirmImpl !== "function") {
        throw new TypeError("confirmImpl must be a function");
    }
    const message = cancelConfirmation(requestId);
    if (!confirmImpl(message)) {
        return Object.freeze({ confirmed: false, response: null });
    }
    const response = await cancelRequest(requestId);
    return Object.freeze({ confirmed: true, response });
}
