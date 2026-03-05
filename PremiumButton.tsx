import { motion } from "framer-motion";

interface PremiumButtonProps {
  children: React.ReactNode;
  className?: string;
}

export default function PremiumButton({
  children,
  className = "",
}: PremiumButtonProps) {
  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", stiffness: 400, damping: 20 }}
      className={`px-6 py-3 rounded-xl bg-zinc-900 text-white font-medium shadow-lg hover:shadow-xl transition-shadow ${className}`}
    >
      {children}
    </motion.button>
  );
}