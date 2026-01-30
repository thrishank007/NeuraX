import QueryInput from "@/components/query/QueryInput";
import ResultsDisplay from "@/components/results/ResultsDisplay";

export default function QueryPage() {
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto pt-12 text-center mb-12">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Insight Engine
          </h1>
          <p className="text-lg text-slate-500 dark:text-slate-400">
            Securely search through your documents, images, and audio files
          </p>
        </div>
        
        <ResultsDisplay />
      </div>
      
      <div className="sticky bottom-0 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent dark:from-slate-900 dark:via-slate-900 pb-8 pt-4">
        <QueryInput />
      </div>
    </div>
  );
}
