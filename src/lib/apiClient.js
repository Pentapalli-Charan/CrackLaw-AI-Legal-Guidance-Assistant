/**
 * CrackLaw API Client
 * Handles all communication with the FastAPI backend.
 */

const API_BASE = '/api/v1';
const HEALTH_BASE = '';

/** Reads optional API key from localStorage */
function getApiKey() {
  return localStorage.getItem('cracklaw_api_key') || '';
}

/** Builds headers including optional API key */
function buildHeaders(extra = {}) {
  const headers = { 'Content-Type': 'application/json', ...extra };
  const apiKey = getApiKey();
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
}

/**
 * Generic fetch wrapper with timeout and error handling.
 */
async function request(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: options.signal || controller.signal,
      headers: buildHeaders(options.headers),
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      let errorBody;
      try {
        errorBody = await response.json();
      } catch {
        errorBody = { message: response.statusText };
      }
      const error = new Error(errorBody.message || `HTTP ${response.status}: ${response.statusText}`);
      error.status = response.status;
      error.body = errorBody;
      throw error;
    }

    return response;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      const error = new Error('Request timed out');
      error.code = 'TIMEOUT';
      throw error;
    }
    if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
      const error = new Error('Unable to connect to the server. Please check if the backend is running.');
      error.code = 'NETWORK_ERROR';
      throw error;
    }
    throw err;
  }
}

/**
 * Send a chat message and receive a complete response.
 * @param {string} sessionId
 * @param {string} query
 * @param {object} [options]
 * @returns {Promise<object>} Full ChatResponse
 */
export async function sendMessage(sessionId, query, options = null) {
  const res = await request(`${API_BASE}/chat`, {
    method: 'POST',
    body: JSON.stringify({ session_id: sessionId, query, options }),
  });
  return res.json();
}

/**
 * Stream a chat message via SSE.
 * @param {string} sessionId
 * @param {string} query
 * @param {object} callbacks - { onToken, onDone, onError }
 * @param {AbortSignal} [signal] - AbortController signal for stop generation
 * @param {object} [options]
 */
export async function streamMessage(sessionId, query, callbacks, signal = null, options = null) {
  const { onToken, onDone, onError } = callbacks;

  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: buildHeaders(),
      body: JSON.stringify({ session_id: sessionId, query, options }),
      signal,
    });

    if (!response.ok) {
      let errorBody;
      try {
        errorBody = await response.json();
      } catch {
        errorBody = { message: response.statusText };
      }
      const error = new Error(errorBody.message || `HTTP ${response.status}`);
      error.status = response.status;
      onError?.(error);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const parts = buffer.split('\n\n');
      // Keep the last incomplete chunk in the buffer
      buffer = parts.pop() || '';

      for (const part of parts) {
        if (!part.trim()) continue;

        let eventName = 'message';
        let dataStr = '';

        for (const line of part.split('\n')) {
          if (line.startsWith('event: ')) {
            eventName = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            dataStr = line.slice(6);
          }
        }

        if (!dataStr) continue;

        try {
          const data = JSON.parse(dataStr);

          if (eventName === 'token') {
            onToken?.(data.token);
          } else if (eventName === 'done') {
            onDone?.(data);
          } else if (eventName === 'error') {
            onError?.(new Error(data.detail || 'Stream error'));
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      // User stopped generation — not an error
      return;
    }
    if (err.message === 'Failed to fetch' || err.message?.includes('NetworkError')) {
      const error = new Error('Connection lost during streaming. Please try again.');
      error.code = 'STREAM_INTERRUPTED';
      onError?.(error);
      return;
    }
    onError?.(err);
  }
}

/**
 * Get chat history for a session.
 * @param {string} sessionId
 * @returns {Promise<object>} HistoryResponse
 */
export async function getHistory(sessionId) {
  const res = await request(`${API_BASE}/chat/history?session_id=${encodeURIComponent(sessionId)}`);
  return res.json();
}

/**
 * Delete chat history for a session.
 * @param {string} sessionId
 * @returns {Promise<object>} HistoryClearResponse
 */
export async function deleteHistory(sessionId) {
  const res = await request(`${API_BASE}/chat/history?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
  return res.json();
}

/**
 * Health check.
 * @returns {Promise<object>}
 */
export async function checkHealth() {
  const res = await request(`${HEALTH_BASE}/health`, {}, 5000);
  return res.json();
}

/**
 * Retry wrapper — retries a function up to N times.
 * @param {Function} fn - Async function to retry
 * @param {number} [retries=2]
 * @param {number} [delayMs=1000]
 */
export async function withRetry(fn, retries = 2, delayMs = 1000) {
  let lastError;
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (i < retries) {
        await new Promise((r) => setTimeout(r, delayMs * (i + 1)));
      }
    }
  }
  throw lastError;
}
