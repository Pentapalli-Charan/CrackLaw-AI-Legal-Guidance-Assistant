/**
 * CrackLaw Chat Store
 * Zustand store managing chat sessions, messages, streaming state, and localStorage persistence.
 */
import { create } from 'zustand';
import * as api from './apiClient';

// ── Helpers ──

function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function loadConversations() {
  try {
    const raw = localStorage.getItem('cracklaw_conversations');
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveConversations(conversations) {
  try {
    localStorage.setItem('cracklaw_conversations', JSON.stringify(conversations));
  } catch {
    // localStorage full or unavailable — silently ignore
  }
}

function loadSessionList() {
  try {
    const raw = localStorage.getItem('cracklaw_sessions');
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveSessionList(sessions) {
  try {
    localStorage.setItem('cracklaw_sessions', JSON.stringify(sessions));
  } catch {
    // silently ignore
  }
}

function loadActiveSession() {
  return localStorage.getItem('cracklaw_active_session') || null;
}

function saveActiveSession(id) {
  if (id) {
    localStorage.setItem('cracklaw_active_session', id);
  } else {
    localStorage.removeItem('cracklaw_active_session');
  }
}

// ── Store ──

const useChatStore = create((set, get) => {
  // Restore persisted state
  const savedConversations = loadConversations();
  const savedSessions = loadSessionList();
  const savedActive = loadActiveSession();

  return {
    // State
    conversations: savedConversations,       // { sessionId: [{ role, content, timestamp, citations?, confidence?, ... }] }
    sessions: savedSessions,                 // [{ id, title, createdAt, updatedAt }]
    activeSessionId: savedActive,
    isStreaming: false,
    error: null,
    abortController: null,

    // ── Session Management ──

    createSession: () => {
      const id = generateSessionId();
      const session = {
        id,
        title: 'New Conversation',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      set((state) => {
        const sessions = [session, ...state.sessions];
        const conversations = { ...state.conversations, [id]: [] };
        saveSessionList(sessions);
        saveConversations(conversations);
        saveActiveSession(id);
        return { sessions, conversations, activeSessionId: id, error: null };
      });

      return id;
    },

    switchSession: (sessionId) => {
      saveActiveSession(sessionId);
      set({ activeSessionId: sessionId, error: null });
    },

    deleteSession: (sessionId) => {
      set((state) => {
        const sessions = state.sessions.filter((s) => s.id !== sessionId);
        const conversations = { ...state.conversations };
        delete conversations[sessionId];

        let activeSessionId = state.activeSessionId;
        if (activeSessionId === sessionId) {
          activeSessionId = sessions.length > 0 ? sessions[0].id : null;
        }

        saveSessionList(sessions);
        saveConversations(conversations);
        saveActiveSession(activeSessionId);

        return { sessions, conversations, activeSessionId, error: null };
      });

      // Fire-and-forget server-side cleanup
      api.deleteHistory(sessionId).catch(() => {});
    },

    // ── Messages ──

    getActiveMessages: () => {
      const { conversations, activeSessionId } = get();
      if (!activeSessionId) return [];
      return conversations[activeSessionId] || [];
    },

    getActiveSession: () => {
      const { sessions, activeSessionId } = get();
      return sessions.find((s) => s.id === activeSessionId) || null;
    },

    /** Send a message and stream the response */
    sendMessage: async (query) => {
      const state = get();
      let sessionId = state.activeSessionId;

      // Auto-create session if none active
      if (!sessionId) {
        sessionId = get().createSession();
      }

      // Add user message
      const userMessage = {
        role: 'user',
        content: query,
        timestamp: Date.now(),
      };

      // Create placeholder assistant message for streaming
      const assistantMessage = {
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        isStreaming: true,
      };

      const abortController = new AbortController();

      set((state) => {
        const messages = [...(state.conversations[sessionId] || []), userMessage, assistantMessage];
        const conversations = { ...state.conversations, [sessionId]: messages };

        // Update session title from first user message
        const sessions = state.sessions.map((s) => {
          if (s.id === sessionId) {
            const title = s.title === 'New Conversation' ? query.slice(0, 50) + (query.length > 50 ? '…' : '') : s.title;
            return { ...s, title, updatedAt: Date.now() };
          }
          return s;
        });

        saveConversations(conversations);
        saveSessionList(sessions);

        return {
          conversations,
          sessions,
          isStreaming: true,
          error: null,
          abortController,
        };
      });

      // Stream response
      await api.streamMessage(
        sessionId,
        query,
        {
          onToken: (token) => {
            set((state) => {
              const messages = [...(state.conversations[sessionId] || [])];
              const lastIdx = messages.length - 1;
              if (lastIdx >= 0 && messages[lastIdx].role === 'assistant') {
                messages[lastIdx] = {
                  ...messages[lastIdx],
                  content: messages[lastIdx].content + token,
                };
                const conversations = { ...state.conversations, [sessionId]: messages };
                // Persist periodically is expensive; we save on done
                return { conversations };
              }
              return {};
            });
          },

          onDone: (data) => {
            set((state) => {
              const messages = [...(state.conversations[sessionId] || [])];
              const lastIdx = messages.length - 1;
              if (lastIdx >= 0 && messages[lastIdx].role === 'assistant') {
                messages[lastIdx] = {
                  ...messages[lastIdx],
                  content: data.response_text || messages[lastIdx].content,
                  isStreaming: false,
                  citations: data.citations || [],
                  confidence_score: data.confidence_score,
                  intent: data.intent,
                  provider: data.provider,
                  model: data.model,
                  structured_data: data.structured_data,
                  latency_ms: data.latency_ms,
                };
                const conversations = { ...state.conversations, [sessionId]: messages };
                saveConversations(conversations);
                return { conversations, isStreaming: false, abortController: null };
              }
              return { isStreaming: false, abortController: null };
            });
          },

          onError: (err) => {
            set((state) => {
              const messages = [...(state.conversations[sessionId] || [])];
              const lastIdx = messages.length - 1;
              if (lastIdx >= 0 && messages[lastIdx].role === 'assistant' && messages[lastIdx].isStreaming) {
                messages[lastIdx] = {
                  ...messages[lastIdx],
                  isStreaming: false,
                  error: err.message || 'An error occurred',
                };
                const conversations = { ...state.conversations, [sessionId]: messages };
                saveConversations(conversations);
                return { conversations, isStreaming: false, error: err.message, abortController: null };
              }
              return { isStreaming: false, error: err.message, abortController: null };
            });
          },
        },
        abortController.signal
      );
    },

    /** Stop the current generation */
    stopGeneration: () => {
      const { abortController } = get();
      if (abortController) {
        abortController.abort();
      }

      set((state) => {
        const sessionId = state.activeSessionId;
        if (!sessionId) return { isStreaming: false, abortController: null };

        const messages = [...(state.conversations[sessionId] || [])];
        const lastIdx = messages.length - 1;
        if (lastIdx >= 0 && messages[lastIdx].role === 'assistant' && messages[lastIdx].isStreaming) {
          messages[lastIdx] = {
            ...messages[lastIdx],
            isStreaming: false,
            stopped: true,
          };
          const conversations = { ...state.conversations, [sessionId]: messages };
          saveConversations(conversations);
          return { conversations, isStreaming: false, abortController: null };
        }
        return { isStreaming: false, abortController: null };
      });
    },

    /** Regenerate the last assistant message */
    regenerate: async () => {
      const state = get();
      const sessionId = state.activeSessionId;
      if (!sessionId) return;

      const messages = [...(state.conversations[sessionId] || [])];

      // Find the last user message
      let lastUserQuery = null;
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user') {
          lastUserQuery = messages[i].content;
          break;
        }
      }
      if (!lastUserQuery) return;

      // Remove the last assistant message
      if (messages.length > 0 && messages[messages.length - 1].role === 'assistant') {
        messages.pop();
      }

      // Also remove the last user message so sendMessage re-adds it
      if (messages.length > 0 && messages[messages.length - 1].role === 'user') {
        messages.pop();
      }

      set((state) => {
        const conversations = { ...state.conversations, [sessionId]: messages };
        saveConversations(conversations);
        return { conversations };
      });

      // Re-send the query
      await get().sendMessage(lastUserQuery);
    },

    /** Retry a failed message */
    retryFailed: async () => {
      return get().regenerate();
    },

    /** Clear error state */
    clearError: () => set({ error: null }),
  };
});

export default useChatStore;
