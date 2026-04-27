"use client";

import { useEffect, useRef, useCallback } from "react";
import { Citation, CitationCandidate } from "@/lib/sse-types";

interface CitationLightboxProps {
  citation: Citation | CitationCandidate;
  index: number;
  isOpen: boolean;
  onClose: () => void;
  triggerRef?: React.RefObject<HTMLElement>;
}

export default function CitationLightbox({
  citation,
  index,
  isOpen,
  onClose,
  triggerRef,
}: CitationLightboxProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  // Handle escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        e.preventDefault();
        onClose();
      }
    },
    [isOpen, onClose]
  );

  // Sync dialog open state
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen) {
      dialog.showModal();
      // Focus the close button when opened
      closeButtonRef.current?.focus();
    } else {
      dialog.close();
    }
  }, [isOpen]);

  // Add/remove keydown listener
  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Return focus to trigger when closing
  useEffect(() => {
    if (!isOpen && triggerRef?.current) {
      triggerRef.current.focus();
    }
  }, [isOpen, triggerRef]);

  if (!isOpen) return null;

  // Extract display properties from either Citation or CitationCandidate
  const imageUri = "image_uri" in citation ? citation.image_uri : "";
  const pageNum = "page_num" in citation ? citation.page_num : 0;
  const docFilename = "doc_filename" in citation ? citation.doc_filename : "Document";
  const sectionPath =
    "section_path" in citation ? citation.section_path : [];
  const extractedText =
    "extracted_text_excerpt" in citation
      ? citation.extracted_text_excerpt
      : "";

  // Build citation string for copy
  const citationText = `[${docFilename}, p.${pageNum}](${imageUri})`;

  const handleCopyCitation = async () => {
    try {
      await navigator.clipboard.writeText(citationText);
      // Could add a toast notification here
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement("textarea");
      textarea.value = citationText;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="fixed inset-0 w-full h-full bg-transparent p-0 m-0 max-w-full max-h-full"
      onClose={onClose}
      aria-labelledby="lightbox-title"
    >
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Lightbox panel */}
      <div
        className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white shadow-xl flex flex-col z-50"
        role="dialog"
        aria-modal="true"
        aria-labelledby="lightbox-title"
      >
        {/* Header */}
        <header className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
          <h2 id="lightbox-title" className="text-lg font-semibold text-gray-900">
            Citation {index}: {docFilename}, p.{pageNum}
          </h2>
          <button
            ref={closeButtonRef}
            onClick={onClose}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Close lightbox"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Section path */}
          {sectionPath.length > 0 && (
            <p className="text-sm text-gray-500 mb-2">
              {sectionPath.join(" > ")}
            </p>
          )}

          {/* Page image */}
          <div className="mb-4 bg-gray-100 rounded-lg overflow-hidden">
            {imageUri ? (
              <img
                src={imageUri}
                alt={`Page ${pageNum} of ${docFilename}`}
                className="w-full h-auto"
              />
            ) : (
              <div className="aspect-[8.5/11] flex items-center justify-center text-gray-400">
                Image not available
              </div>
            )}
          </div>

          {/* Extracted text */}
          {extractedText && (
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">
                Extracted Text
              </h3>
              <p className="text-sm text-gray-600 whitespace-pre-wrap">
                {extractedText}
              </p>
            </div>
          )}

          {/* Document info */}
          <div className="text-sm text-gray-500">
            <p>
              <span className="font-medium">Document:</span> {docFilename}
            </p>
            <p>
              <span className="font-medium">Page:</span> {pageNum}
            </p>
          </div>
        </div>

        {/* Footer with actions */}
        <footer className="p-4 border-t border-gray-200 bg-white">
          <button
            onClick={handleCopyCitation}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Copy citation
          </button>
        </footer>
      </div>
    </dialog>
  );
}
