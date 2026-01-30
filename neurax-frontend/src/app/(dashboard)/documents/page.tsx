import FileUploader from "@/components/upload/FileUploader";
import FileList from "@/components/upload/FileList";
import { Files } from "lucide-react";

export default function DocumentsPage() {
  return (
    <div className="space-y-8 pb-12">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white flex items-center">
            <Files className="mr-3 text-blue-500" /> Document Library
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Manage your knowledge base and processing tasks
          </p>
        </div>
      </div>

      <section>
        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">
          Upload New Documents
        </h2>
        <FileUploader />
      </section>

      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">
            Processed Files
          </h2>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400">Filter by:</span>
            <select className="bg-transparent text-xs font-medium text-slate-600 dark:text-slate-300 border-0 focus:ring-0 cursor-pointer">
              <option>All Types</option>
              <option>Documents</option>
              <option>Images</option>
              <option>Audio</option>
            </select>
          </div>
        </div>
        <FileList />
      </section>
    </div>
  );
}
