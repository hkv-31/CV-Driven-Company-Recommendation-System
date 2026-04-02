import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    <div className="w-64 h-screen bg-gray-100 dark:bg-zinc-950 border-r border-gray-200 dark:border-zinc-800 p-8 flex flex-col">

      <h1 className="text-xl font-bold mb-12 text-black dark:text-white">
        HAA
      </h1>

      <nav className="flex flex-col gap-6">

        <Link
          to="/"
          className="text-gray-700 dark:text-zinc-400 hover:text-black dark:hover:text-white transition"
        >
          Overview
        </Link>

        <Link
          to="/recommendations"
          className="text-gray-700 dark:text-zinc-400 hover:text-black dark:hover:text-white transition"
        >
          Companies
        </Link>

        <Link
          to="/upload"
          className="text-gray-700 dark:text-zinc-400 hover:text-black dark:hover:text-white transition"
        >
          Upload Resume
        </Link>

        <Link
          to="/account"
          className="text-gray-700 dark:text-zinc-400 hover:text-black dark:hover:text-white transition"
        >
          Account
        </Link>

      </nav>

    </div>
  );
}
