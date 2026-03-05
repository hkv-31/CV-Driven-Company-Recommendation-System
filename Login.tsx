import { useState } from "react";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  return (
    <div className="flex justify-center items-center py-24">
      <div className="bg-white border border-zinc-200 p-12 rounded-3xl shadow-lg w-full max-w-md">
        <h2 className="text-3xl font-semibold text-zinc-900 mb-8 text-center">
          Sign In
        </h2>

        <input
          type="email"
          placeholder="Email"
          className="w-full mb-5 px-4 py-3 rounded-xl border border-zinc-200 focus:ring-2 focus:ring-zinc-400 outline-none"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        <input
          type="password"
          placeholder="Password"
          className="w-full mb-8 px-4 py-3 rounded-xl border border-zinc-200 focus:ring-2 focus:ring-zinc-400 outline-none"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        <button className="w-full bg-zinc-900 text-white py-3 rounded-xl hover:bg-zinc-800 transition">
          Continue
        </button>
      </div>
    </div>
  );
}