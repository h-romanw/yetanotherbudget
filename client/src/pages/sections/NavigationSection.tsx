import { BarChartIcon, TableIcon, TargetIcon, UploadIcon } from "lucide-react";
import React from "react";
import { Button } from "@/components/ui/button";

const navigationItems = [
  {
    icon: TableIcon,
    label: "Import & View",
    isActive: true,
  },
  {
    icon: BarChartIcon,
    label: "Analyze & Understand",
    isActive: false,
  },
  {
    icon: TargetIcon,
    label: "Set Targets (WIP)",
    isActive: false,
  },
];

export const NavigationSection = (): JSX.Element => {
  return (
    <nav className="flex flex-col w-64 max-w-64 items-start gap-20 p-4 bg-[#e2999b]">
      <header className="flex h-[72px] items-center gap-4 px-0 py-0.5 w-full">
        <img
          className="w-12 h-12 object-cover"
          alt="Cash"
          src="/figmaAssets/cash-5816752-1.png"
        />

        <div className="flex flex-col items-start justify-center gap-2 flex-1">
          <h1 className="font-heading-4 font-[number:var(--heading-4-font-weight)] text-black text-[length:var(--heading-4-font-size)] tracking-[var(--heading-4-letter-spacing)] leading-[var(--heading-4-line-height)] whitespace-nowrap [font-style:var(--heading-4-font-style)]">
            Yet Another Budget
          </h1>

          <p className="font-body font-[number:var(--body-font-weight)] text-black text-[length:var(--body-font-size)] tracking-[var(--body-letter-spacing)] leading-[var(--body-line-height)] whitespace-nowrap [font-style:var(--body-font-style)]">
            Your spending, analyzed by AI.
          </p>
        </div>
      </header>

      <div className="flex flex-col items-start justify-center gap-4 w-full">
        {navigationItems.map((item, index) => {
          const IconComponent = item.icon;
          return (
            <Button
              key={index}
              variant="ghost"
              className={`flex h-16 items-center justify-start gap-2 p-2 w-full rounded-[10px] overflow-hidden h-auto ${
                item.isActive
                  ? "bg-[#52181e] hover:bg-[#52181e]"
                  : "bg-transparent hover:bg-[#d88a8c]"
              }`}
            >
              <IconComponent
                className={`w-10 h-10 ${
                  item.isActive ? "text-[#f8f8f8]" : "text-[#050101]"
                }`}
              />

              <span
                className={`font-heading-5 font-[number:var(--heading-5-font-weight)] text-[length:var(--heading-5-font-size)] tracking-[var(--heading-5-letter-spacing)] leading-[var(--heading-5-line-height)] whitespace-nowrap [font-style:var(--heading-5-font-style)] ${
                  item.isActive ? "text-[#f8f8f8]" : "text-[#050101]"
                }`}
              >
                {item.label}
              </span>
            </Button>
          );
        })}
      </div>

      <div className="flex flex-col items-center justify-end gap-2 px-0 py-2 flex-1 w-full">
        <div className="w-full h-[491px]" />

        <div className="w-[232px] h-[72px] bg-[#260a0c] rounded-[20px] overflow-hidden">
          <Button className="flex w-[226px] h-[61px] items-center justify-center gap-2 px-0 py-4 relative top-1 left-1 bg-[#832632] rounded-[20px] overflow-hidden hover:bg-[#9a2d3b] h-auto">
            <UploadIcon className="w-6 h-6 text-[#d7d7d7]" />

            <span className="mt-[-3.5px] mb-[-1.5px] font-heading-5 font-[number:var(--heading-5-font-weight)] text-[#d7d7d7] text-[length:var(--heading-5-font-size)] tracking-[var(--heading-5-letter-spacing)] leading-[var(--heading-5-line-height)] whitespace-nowrap [font-style:var(--heading-5-font-style)]">
              Import new data
            </span>
          </Button>
        </div>
      </div>
    </nav>
  );
};
