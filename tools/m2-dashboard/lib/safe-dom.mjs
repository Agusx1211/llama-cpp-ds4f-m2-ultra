const REDACTED = "[redacted]";

export function setSafeText(element, value) {
    element.textContent = value === null || value === undefined ? "" : String(value);
    return element;
}

export function contentForDisplay(content, { allowReveal = false, revealed = false } = {}) {
    if (!allowReveal || !revealed) {
        return REDACTED;
    }
    return String(content ?? "");
}

export function formatBytes(value) {
    if (!Number.isFinite(value) || value < 0) {
        return "—";
    }
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let scaled = value;
    let unit = 0;
    while (scaled >= 1024 && unit < units.length - 1) {
        scaled /= 1024;
        unit += 1;
    }
    const digits = unit === 0 || scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
    return `${scaled.toFixed(digits)} ${units[unit]}`;
}

export function formatDuration(value) {
    if (!Number.isFinite(value) || value < 0) {
        return "—";
    }
    if (value < 1000) {
        return `${Math.round(value)} ms`;
    }
    if (value < 60_000) {
        return `${(value / 1000).toFixed(1)} s`;
    }
    return `${(value / 60_000).toFixed(1)} min`;
}

export function formatRate(value, suffix = "/s") {
    if (!Number.isFinite(value) || value < 0) {
        return "—";
    }
    return `${value.toFixed(value >= 100 ? 0 : 1)}${suffix}`;
}
