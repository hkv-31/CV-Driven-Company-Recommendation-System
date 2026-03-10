import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="border-b border-zinc-200 bg-white">
      <div className="max-w-6xl mx-auto px-6 py-5 flex justify-between items-center">
        <h1 className="text-xl font-semibold tracking-tight text-zinc-800">
          Career Recommendation System
        </h1>

        <div className="space-x-8 text-sm font-medium text-zinc-600">
          <Link className="hover:text-zinc-900 transition" to="/">Home</Link>
          <Link className="hover:text-zinc-900 transition" to="/recommendations">Companies</Link>
          <Link className="hover:text-zinc-900 transition" to="/upload">Upload</Link>
          <Link
            className="px-4 py-2 rounded-lg bg-zinc-900 text-white hover:bg-zinc-800 transition"
            to="/login"
          >
            Login
          </Link>
        </div>
      </div>
    </nav>
  );
}
