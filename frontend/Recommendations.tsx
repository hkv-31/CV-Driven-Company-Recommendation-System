import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const companies = [
  { id: 1, name: "Google", description: "AI & Search Technology" },
  { id: 2, name: "Microsoft", description: "Cloud Infrastructure" },
  {id: 3, name: "Amazon", description: "Customer Service Associate"},
  {id: 4, name: "Apple", description: "Genius/Technical Specialist"}
];

export default function Recommendations() {
  return (
    <div>
      <h1 className="text-4xl font-semibold text-zinc-900 mb-12">
        Companies
      </h1>

      <div className="space-y-6">
        {companies.map((company, index) => (
          <motion.div
            key={company.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ y: -6 }}
            className="bg-white p-8 rounded-2xl border border-zinc-200 shadow-sm hover:shadow-xl transition"
          >
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-xl font-semibold text-zinc-900">
                  {company.name}
                </h3>
                <p className="text-zinc-500 mt-2">
                  {company.description}
                </p>
              </div>

              <Link
                to={`/company/${company.id}`}
                className="text-sm font-medium text-zinc-900 hover:underline"
              >
                View →
              </Link>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
