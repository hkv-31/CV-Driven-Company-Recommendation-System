export default function Landing() {
  return (
    <div className="grid grid-cols-3 gap-10">

      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow">
        <p className="text-gray-500 dark:text-zinc-400">Matches Found</p>
        <h2 className="text-3xl font-bold text-black dark:text-white">0</h2>
      </div>

      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow">
        <p className="text-gray-500 dark:text-zinc-400">Applications Sent</p>
        <h2 className="text-3xl font-bold text-black dark:text-white">0</h2>
      </div>

      <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow">
        <p className="text-gray-500 dark:text-zinc-400">Profile Strength</p>
        <h2 className="text-3xl font-bold text-black dark:text-white">0%</h2>
      </div>

    </div>
  );
}