export default function UploadResume() {
  return (
    <div className="flex justify-center py-20">
      <div className="bg-white p-10 rounded-2xl shadow-lg w-full max-w-lg text-center">
        <h2 className="text-2xl font-bold mb-6">
          Upload Your Resume
        </h2>

        <input
          type="file"
          className="w-full mb-6 p-3 border rounded-lg"
        />

        <button className="bg-black text-white px-6 py-3 rounded-lg00 text-white px-6 py-3 rounded-lg hover:bg-zinc-800 transition">
          Analyze Resume
        </button>
      </div>
    </div>
  );
}
