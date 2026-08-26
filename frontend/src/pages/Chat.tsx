import PageTransition from "../components/navigation/PageTransition";
import ChatInterface from "../components/chat/ChatInterface";

export default function Chat() {
  return (
    <PageTransition>
      <ChatInterface />
    </PageTransition>
  );
}
