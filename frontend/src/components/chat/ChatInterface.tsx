import { useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Camera, Mic, SendHorizontal } from "lucide-react";

type Message = {
  id: number;
  role: "user" | "bot";
  text: string;
};

const WELCOME =
  "Hey — I'm the TrendLens bot. Ask me what's trending in food, fashion, photography or beauty right now, or upload a photo and I'll tell you what aesthetic I see.";

const MOCK_REPLIES: Array<[RegExp, string]> = [
  [
    /trend|rising|hot|popular|emerging|aesthetic|viral/i,
    "Right now the strongest risers in the index are minimalist latte art (steady climb for ~10 days) and rustic brunch spreads (small base, fast growth). Both are still mostly unnamed — which is exactly why they're interesting.",
  ],
  [
    /food|cafe|coffee|latte|dessert|brunch|pastry/i,
    "In food, the winning grammar this window: warm natural light, hands-in-frame action, forty-five degree table angles. The highest-engagement look pairs a single hero object with an uncluttered surface.",
  ],
  [
    /fashion|outfit|style|streetwear/i,
    "Fashion clusters are quieter this week, but layered neutral outfits with textured fabrics (linen, raw denim) keep growing. Vintage/thrifted looks hold steady — that one stopped being 'emerging' about six months ago.",
  ],
  [
    /photo|photography|camera|light/i,
    "Photography side: film-grain looks and high-contrast street scenes at night are both rising. Natural light dominates food; hard flash is creeping back into party content.",
  ],
];

function mockReply(query: string): string {
  for (const [pattern, reply] of MOCK_REPLIES) {
    if (pattern.test(query)) return reply;
  }
  return "Interesting angle. My index covers visual trends across food, fashion, photography and beauty — ask me what's rising in any of those, or describe a look you keep seeing and I'll tell you if it's clustered yet.";
}

async function fetchBotReply(query: string): Promise<{ text: string; live: boolean }> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 9000);
    const res = await fetch("/api/rag-query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();
    if (!data?.answer || data?.inScope === false) {
      return {
        text:
          typeof data?.answer === "string"
            ? data.answer
            : mockReply(query),
        live: Boolean(data?.answer),
      };
    }
    return { text: String(data.answer), live: true };
  } catch {
    return { text: mockReply(query), live: false };
  }
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 0, role: "bot", text: WELCOME },
  ]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [attachment, setAttachment] = useState<string | null>(null);
  const [listening, setListening] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const nextId = useRef(1);
  const reduce = useReducedMotion();

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      listRef.current?.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: reduce ? "auto" : "smooth",
      });
    });
  };

  const push = (msg: Omit<Message, "id">) => {
    setMessages((prev) => [...prev, { ...msg, id: nextId.current++ }]);
    scrollToBottom();
  };

  const send = async () => {
    const text = input.trim();
    if ((!text && !attachment) || typing) return;
    const shown = attachment ? `[photo attached] ${text}` : text;
    push({ role: "user", text: shown.trim() });
    setInput("");
    const hadAttachment = attachment;
    setAttachment(null);
    setTyping(true);

    let result: { text: string; live: boolean };
    if (hadAttachment) {
      await new Promise((r) => setTimeout(r, 1100));
      result = {
        text: "I can see the photo you attached. Visual understanding lands here soon — meanwhile, describe it in a few words (e.g. 'latte art on a wooden table') and I'll check whether it's part of any rising cluster.",
        live: false,
      };
    } else {
      result = await fetchBotReply(text);
    }

    setTyping(false);
    push({ role: "bot", text: result.text });
  };

  const onPickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAttachment(file.name);
    e.target.value = "";
  };

  const toggleMic = () => {
    setListening((v) => !v);
    window.setTimeout(() => setListening(false), 2200);
  };

  return (
    <div className="flex h-[calc(100svh-4rem)] flex-col pt-16 md:h-screen">
      {/* messages */}
      <div
        ref={listRef}
        aria-live="polite"
        className="flex-1 space-y-5 overflow-y-auto px-5 py-8 md:px-10"
      >
        <div className="mx-auto max-w-2xl space-y-5">
          <AnimatePresence initial={false}>
            {messages.map((m) => (
              <motion.div
                key={m.id}
                initial={{ opacity: 0, y: reduce ? 0 : 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 26 }}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-5 py-3.5 text-sm leading-relaxed md:text-base ${
                    m.role === "user"
                      ? "rounded-br-sm bg-ink text-paper"
                      : "rounded-bl-sm border border-line bg-paper-deep text-ink"
                  }`}
                >
                  {m.text}
                </div>
              </motion.div>
            ))}
            {typing && (
              <motion.div
                key="typing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex justify-start"
              >
                <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm border border-line bg-paper-deep px-5 py-4">
                  {[0, 1, 2].map((i) => (
                    <motion.span
                      key={i}
                      animate={{ y: [0, -4, 0], opacity: [0.4, 1, 0.4] }}
                      transition={{
                        duration: 0.9,
                        repeat: Infinity,
                        delay: i * 0.15,
                      }}
                      className="h-1.5 w-1.5 rounded-full bg-ink-soft"
                    />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* composer */}
      <div className="border-t border-line px-5 pb-6 pt-4 md:px-10">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void send();
          }}
          className="mx-auto max-w-2xl"
        >
          {attachment && (
            <p className="mb-2 inline-flex items-center gap-2 rounded-full bg-accent-soft px-3 py-1 text-xs text-ink">
              Attached: {attachment}
              <button
                type="button"
                onClick={() => setAttachment(null)}
                aria-label="Remove attachment"
                className="font-semibold text-accent"
              >
                ×
              </button>
            </p>
          )}
          <div className="flex items-end gap-2 rounded-2xl border border-line bg-paper-deep p-2 focus-within:border-ink">
            <label
              htmlFor="chat-camera"
              className="flex h-10 w-10 shrink-0 cursor-pointer items-center justify-center rounded-full text-ink-soft transition-colors hover:bg-line/50 hover:text-ink"
              title="Attach a photo"
            >
              <Camera className="h-5 w-5" aria-hidden />
              <span className="sr-only">Attach a photo</span>
            </label>
            <input
              id="chat-camera"
              ref={fileRef}
              type="file"
              accept="image/*"
              onChange={onPickFile}
              className="hidden"
            />
            <button
              type="button"
              onClick={toggleMic}
              aria-pressed={listening}
              title="Voice input (coming soon)"
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full transition-colors hover:bg-line/50 ${
                listening ? "text-accent" : "text-ink-soft hover:text-ink"
              }`}
            >
              <Mic
                className={`h-5 w-5 ${listening && !reduce ? "animate-pulse" : ""}`}
                aria-hidden
              />
              <span className="sr-only">Voice input</span>
            </button>
            <label htmlFor="chat-input" className="sr-only">
              Message
            </label>
            <textarea
              id="chat-input"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Ask about a trend…"
              className="max-h-32 flex-1 resize-none bg-transparent py-2.5 text-sm focus:outline-none md:text-base"
            />
            <button
              type="submit"
              disabled={!input.trim() && !attachment}
              aria-label="Send message"
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-ink text-paper transition-transform enabled:hover:-translate-y-0.5 disabled:opacity-30"
            >
              <SendHorizontal className="h-4.5 w-4.5" aria-hidden />
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-ink-soft">
            Answers come from the TrendLens pipeline when it's running — otherwise from demo data.
          </p>
        </form>
      </div>
    </div>
  );
}
