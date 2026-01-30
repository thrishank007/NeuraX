"use client";

import { useState, useRef } from "react";
import { Send, Image as ImageIcon, Mic, X } from "lucide-react";
import { useQueryStore } from "@/lib/store/queryStore";
import { queryApi } from "@/lib/api/client";

export default function QueryInput() {
  const { currentQuery, setCurrentQuery, setIsQuerying, setResults, addToHistory } = useQueryStore();
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const removeImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentQuery.trim() && !selectedImage) return;

    setIsQuerying(true);
    try {
      const result = await queryApi.submit(currentQuery, selectedImage || undefined);
      setResults(result);
      addToHistory(currentQuery, result);
    } catch (error) {
      console.error("Query failed", error);
      // Handle error
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <form onSubmit={handleSubmit} className="relative">
        <div className="overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm transition-all focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-500">
          {imagePreview && (
            <div className="p-4 border-b dark:border-slate-800">
              <div className="relative inline-block">
                <img 
                  src={imagePreview} 
                  alt="Preview" 
                  className="h-20 w-20 object-cover rounded-lg border dark:border-slate-700" 
                />
                <button
                  type="button"
                  onClick={removeImage}
                  className="absolute -top-2 -right-2 p-1 bg-white dark:bg-slate-800 border dark:border-slate-700 rounded-full shadow-sm text-slate-500 hover:text-red-500"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          )}
          
          <div className="flex items-end p-2 px-4">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="p-2 text-slate-500 hover:text-blue-500 transition-colors"
              title="Upload image"
            >
              <ImageIcon size={20} />
            </button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageChange}
              accept="image/*"
              className="hidden"
            />
            
            <textarea
              rows={1}
              value={currentQuery}
              onChange={(e) => setCurrentQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask anything about your documents..."
              className="flex-1 max-h-48 min-h-[44px] bg-transparent border-0 focus:ring-0 py-3 px-2 text-slate-800 dark:text-slate-100 placeholder:text-slate-400 resize-none overflow-y-auto"
            />
            
            <button
              type="button"
              className="p-2 text-slate-500 hover:text-blue-500 transition-colors"
              title="Voice input"
            >
              <Mic size={20} />
            </button>
            
            <button
              type="submit"
              disabled={!currentQuery.trim() && !selectedImage}
              className="ml-2 p-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 dark:disabled:bg-slate-800 text-white rounded-xl transition-all shadow-sm shadow-blue-500/20"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
        <p className="mt-2 text-center text-xs text-slate-400">
          Multimodal search powered by CLIP and local LLMs
        </p>
      </form>
    </div>
  );
}
