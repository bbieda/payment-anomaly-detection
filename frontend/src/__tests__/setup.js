// jsdom nie ma ResizeObserver (wymaga go Recharts).
globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
