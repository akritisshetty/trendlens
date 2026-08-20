import { ChatMessage, Conversation, AttachedFile } from '../types';

/**
 * TrendLens API Service Layer
 * Routes all user queries through /api/rag-query — the FAISS cluster retrieval endpoint.
 * No LLM is used. Responses are structured Markdown formatted directly from cluster metadata.
 *
 * Scope restriction: TrendLens only answers social media visual trend questions.
 * Off-topic queries are rejected by the backend with a clear scope message.
 */

const STORAGE_KEY = 'trendlens_conversations_v1';

// Initial starter conversations — trend-focused examples
const DEFAULT_CONVERSATIONS: Conversation[] = [
  {
    id: 'conv-default-1',
    title: 'Food Photography Trends',
    createdAt: new Date(Date.now() - 3600000 * 2).toISOString(),
    updatedAt: new Date(Date.now() - 3600000 * 2).toISOString(),
    messages: [
      {
        id: 'msg-1',
        sender: 'user',
        content: 'What visual food photography trends are rising right now?',
        timestamp: '10:14 AM'
      },
      {
        id: 'msg-2',
        sender: 'assistant',
        content: `Ask TrendLens about any social media visual trend. Examples:

- **"What rising food photography styles get the highest engagement?"**
- **"I'm a fashion creator — what background and lighting is trending?"**
- **"What are the most viral nature photography aesthetics?"**
- **"Show me declining travel content styles to avoid."**

All answers come directly from the TrendLens FAISS cluster database — 5,000 sampled images from the SMPD dataset, no LLM involved.`,
        timestamp: '10:15 AM'
      }
    ]
  },
  {
    id: 'conv-default-2',
    title: 'Fashion & Style Aesthetics',
    createdAt: new Date(Date.now() - 3600000 * 24).toISOString(),
    updatedAt: new Date(Date.now() - 3600000 * 24).toISOString(),
    messages: [
      {
        id: 'msg-3',
        sender: 'user',
        content: 'What visual aesthetics are trending in fashion photography?',
        timestamp: 'Yesterday'
      },
      {
        id: 'msg-4',
        sender: 'assistant',
        content: `TrendLens is ready to answer your visual trend question. Use the input box below to query the FAISS cluster database for:

- Fashion, food, travel, nature, nightlife, street, architecture photography trends
- Engagement rates and viral patterns per cluster
- Rising vs Declining lifecycle stages
- Creator strategy: composition, lighting, colour palette, props`,
        timestamp: 'Yesterday'
      }
    ]
  }
];

export const chatService = {
  /**
   * Fetch all stored conversations
   */
  async getChats(): Promise<Conversation[]> {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_CONVERSATIONS));
        return DEFAULT_CONVERSATIONS;
      }
      return JSON.parse(stored);
    } catch (e) {
      console.error('Error reading conversations from storage:', e);
      return DEFAULT_CONVERSATIONS;
    }
  },

  /**
   * Get single conversation by ID
   */
  async getChatById(id: string): Promise<Conversation | null> {
    const chats = await this.getChats();
    return chats.find((c) => c.id === id) || null;
  },

  /**
   * Create a new conversation thread
   */
  async createChat(initialTitle: string = 'New Conversation'): Promise<Conversation> {
    const chats = await this.getChats();
    const newConv: Conversation = {
      id: `conv-${Date.now()}`,
      title: initialTitle,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      messages: []
    };
    const updated = [newConv, ...chats];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    return newConv;
  },

  /**
   * Save or update conversation state
   */
  async saveChat(conversation: Conversation): Promise<void> {
    const chats = await this.getChats();
    const idx = chats.findIndex((c) => c.id === conversation.id);
    let updated: Conversation[];
    if (idx >= 0) {
      updated = [...chats];
      updated[idx] = { ...conversation, updatedAt: new Date().toISOString() };
    } else {
      updated = [conversation, ...chats];
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  },

  /**
   * Delete a conversation by ID
   */
  async deleteChat(id: string): Promise<void> {
    const chats = await this.getChats();
    const updated = chats.filter((c) => c.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  },

  /**
   * Rename a conversation thread
   */
  async renameChat(id: string, newTitle: string): Promise<void> {
    const chats = await this.getChats();
    const updated = chats.map((c) => (c.id === id ? { ...c, title: newTitle } : c));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  },

  /**
   * Send a user message and receive a FAISS cluster intelligence response.
   * Routes to /api/rag-query — no LLM, grounded in real cluster metadata.
   */
  async sendMessage(
    messageText: string,
    history: ChatMessage[] = [],
    attachments: AttachedFile[] = []
  ): Promise<ChatMessage> {
    try {
      // Route all messages through /api/rag-query — FAISS cluster retrieval, no LLM.
      const response = await fetch('/api/rag-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: messageText })
      });

      if (response.ok) {
        const data = await response.json();
        return {
          id: `msg-${Date.now()}`,
          sender: 'assistant',
          content: data.answer || 'TrendLens could not find relevant clusters for this query.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          inScope: data.inScope,
          scopeReason: data.scopeReason ?? null,
          scopeMethod: data.scopeMethod ?? null,
          retrievedClusters: data.retrievedClusters || [],
          supportingImages: data.supportingImages || []
        };
      }
    } catch (e) {
      console.warn('TrendLens API unavailable:', e);
    }

    // Graceful fallback when server is unreachable
    return {
      id: `msg-${Date.now()}`,
      sender: 'assistant',
      content: [
        '⚠️ **TrendLens backend unreachable**',
        '',
        'Could not connect to the TrendLens backend. The frontend only serves real pipeline data (no fabricated results).',
        'Start both servers:',
        '```',
        'cd trendlens && source venv/bin/activate && python -m src.api   # Python backend :8000',
        'cd frontend && npx tsx server.ts                                 # React frontend :3000',
        '```',
        '',
        'TrendLens tracks emerging visual trends from real social media data — early signals before they go mainstream.',
      ].join('\n'),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
  },

  /**
   * Upload files placeholder
   */
  async uploadFiles(files: File[]): Promise<AttachedFile[]> {
    return files.map((file, idx) => {
      const isImg = file.type.startsWith('image/');
      const isAud = file.type.startsWith('audio/');
      const isData = file.name.endsWith('.csv') || file.name.endsWith('.json') || file.name.endsWith('.xlsx');
      
      let category: AttachedFile['category'] = 'document';
      if (isImg) category = 'image';
      else if (isAud) category = 'audio';
      else if (isData) category = 'data';

      return {
        id: `att-${Date.now()}-${idx}`,
        name: file.name,
        size: file.size,
        type: file.type || 'application/octet-stream',
        url: URL.createObjectURL(file),
        fileObject: file,
        progress: 100,
        category
      };
    });
  },

  /**
   * Text to Speech helper
   */
  async textToSpeech(text: string): Promise<void> {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text.replace(/[*#`_]/g, ''));
      utterance.rate = 0.95;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  },

  /**
   * Analyze image placeholder — NO visual analysis model is implemented.
   * Returns an honest message instead of fabricating CLIP/BLIP results.
   */
  async analyzeImage(file: File): Promise<string> {
    return [
      '⚠️ **Image analysis is not implemented in this build.**',
      '',
      `Received \`${file.name}\` but TrendLens has no image-understanding model wired to the chat frontend.`,
      'The backend pipeline can caption cluster representatives, but single-image analysis is NOT EVALUATED.',
      '',
      'You can still ask text questions about **social media visual trends** (e.g. "dog photography style").',
    ].join('\n');
  }
};
