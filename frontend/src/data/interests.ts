import {
  UtensilsCrossed,
  Shirt,
  Palette,
  Music,
  Cpu,
  PenTool,
  Plane,
  Dribbble,
  Clapperboard,
  BookOpen,
  Camera,
  Gamepad2,
  Globe2,
  FlaskConical,
  type LucideIcon,
} from "lucide-react";

export type Interest = {
  id: string;
  label: string;
  icon: LucideIcon;
};

export const INTERESTS: Interest[] = [
  { id: "food", label: "Food", icon: UtensilsCrossed },
  { id: "fashion", label: "Fashion", icon: Shirt },
  { id: "art", label: "Art", icon: Palette },
  { id: "music", label: "Music", icon: Music },
  { id: "technology", label: "Technology", icon: Cpu },
  { id: "design", label: "Design", icon: PenTool },
  { id: "travel", label: "Travel", icon: Plane },
  { id: "sports", label: "Sports", icon: Dribbble },
  { id: "movies", label: "Movies", icon: Clapperboard },
  { id: "books", label: "Books", icon: BookOpen },
  { id: "photography", label: "Photography", icon: Camera },
  { id: "gaming", label: "Gaming", icon: Gamepad2 },
  { id: "culture", label: "Culture", icon: Globe2 },
  { id: "science", label: "Science", icon: FlaskConical },
];
