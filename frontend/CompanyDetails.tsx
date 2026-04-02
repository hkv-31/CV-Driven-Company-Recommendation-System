import { useParams } from "react-router-dom";
import PremiumButton from "../components/PremiumButton";

const companies = [
  {
    id: 1,
    name: "Google",
    description: "AI & Search Technology",
    hiringPeriod: "Jan - March",
    applyLink: "https://careers.google.com",
  },
  {
    id: 2,
    name: "Microsoft",
    description: "Cloud Infrastructure",
    hiringPeriod: "Feb - April",
    applyLink: "https://careers.microsoft.com",
  },
  {
    id: 3,
    name: "Amazon",
    description: "Customer Service Associate",
    hiringPeriod: "Jan - March",
    applyLink: "https://careers.amazon.com",
  },
  {
    id: 4,
    name: "Apple",
    description: "Genius/Technical Specialist",
    hiringPeriod: "Jan - March",
    applyLink: "https://careers.apple.com",
  }
];

export default function CompanyDetails() {
  const { id } = useParams();
  const company = companies.find((c) => c.id === Number(id));

  if (!company) return <h2>Not Found</h2>;

  return (
    <div className="bg-white p-12 rounded-3xl border border-zinc-200 shadow-sm max-w-3xl">
      <h1 className="text-4xl font-semibold text-zinc-900 mb-6">
        {company.name}
      </h1>

      <p className="text-zinc-600 mb-8">
        {company.description}
      </p>

      <div className="text-sm text-zinc-500 mb-10">
        Hiring Period: {company.hiringPeriod}
      </div>

      <a
  href={company.applyLink}
  target="_blank"
  rel="noopener noreferrer"
>
  <PremiumButton>
    Apply Now
  </PremiumButton>
  </a>
    </div>
  );
}
