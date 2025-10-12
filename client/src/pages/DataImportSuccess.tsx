import React from "react";
import { NavigationSection } from "./sections/NavigationSection";
import { SummarySection } from "./sections/SummarySection";

export const DataImportSuccess = (): JSX.Element => {
  return (
    <div className="flex w-full bg-[#d7d7d7]">
      <NavigationSection />
      <SummarySection />
    </div>
  );
};
