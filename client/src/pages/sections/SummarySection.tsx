import React from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";

const transactionData = [
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "RENT",
    value: "£1776.50",
    category: "BILLS & UTILITIES",
    categoryColor: "bg-[#d34dd5]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
  {
    date: "12/10/2025",
    payee: "TESCO STORES",
    value: "£35.64",
    category: "Groceries",
    categoryColor: "bg-[#19b36b]",
  },
];

export const SummarySection = (): JSX.Element => {
  return (
    <section className="flex flex-col w-full items-start gap-4">
      <div className="flex items-start gap-4 w-full">
        <Card className="flex flex-col flex-1 items-start justify-between bg-[#f8f8f8] rounded-[20px] overflow-hidden border-0">
          <CardContent className="flex flex-col w-full h-full p-0">
            <header className="flex h-12 items-end gap-6 px-4 py-0 w-full">
              <div className="inline-flex items-center gap-2">
                <div className="inline-flex h-10 items-center px-2 py-0">
                  <span className="font-[number:var(--small-font-weight)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] font-small [font-style:var(--small-font-style)] text-[length:var(--small-font-size)] text-[#020202]">
                    BALANCE
                  </span>
                </div>

                <div className="inline-flex h-10 items-center px-2 py-0">
                  <span className="font-[number:var(--body-font-weight)] tracking-[var(--body-letter-spacing)] leading-[var(--body-line-height)] font-body [font-style:var(--body-font-style)] text-[length:var(--body-font-size)] text-[#020202]">
                    £2,957.00
                  </span>
                </div>
              </div>
            </header>

            <div className="flex flex-col items-center justify-center gap-2 flex-1 w-full">
              <div className="relative w-[480px] h-[480px] max-w-full bg-[#f6f6f6]">
                <img
                  className="absolute w-[100.00%] h-[100.00%] top-[-17.44%] left-[-17.44%]"
                  alt="Vector"
                  src="/figmaAssets/vector.svg"
                />

                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-1">
                    <span className="font-heading-3 font-[number:var(--heading-3-font-weight)] text-[#5f6464] text-[length:var(--heading-3-font-size)] text-center tracking-[var(--heading-3-letter-spacing)] leading-[var(--heading-3-line-height)] [font-style:var(--heading-3-font-style)]">
                      Spent
                    </span>

                    <span className="font-heading-2 font-[number:var(--heading-2-font-weight)] text-[#020202] text-[length:var(--heading-2-font-size)] text-center tracking-[var(--heading-2-letter-spacing)] leading-[var(--heading-2-line-height)] [font-style:var(--heading-2-font-style)]">
                      £2,750
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="flex flex-col flex-1 items-start justify-between bg-[#f8f8f8] rounded-[20px] overflow-hidden border-0">
          <CardContent className="flex flex-col w-full h-full p-0">
            <header className="flex items-start gap-6 px-2 py-0 w-full bg-[#f8f8f8]">
              <div className="inline-flex items-center gap-2">
                <div className="inline-flex h-10 items-center px-2 py-0">
                  <span className="font-[number:var(--small-font-weight)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] font-small [font-style:var(--small-font-style)] text-[length:var(--small-font-size)] text-[#020202]">
                    SUMMARY
                  </span>
                </div>
              </div>
            </header>

            <div className="flex flex-col items-start gap-2 px-4 py-8 flex-1 w-full">
              <ScrollArea className="flex-1 w-full">
                <p className="font-body font-[number:var(--body-font-weight)] text-black text-[length:var(--body-font-size)] tracking-[var(--body-letter-spacing)] leading-[var(--body-line-height)] [font-style:var(--body-font-style)]">
                  Based on your debit card statement, your spending patterns
                  show a strong emphasis on essentials and routine purchases,
                  with consistent transactions in categories like groceries,
                  transport, and recurring bills. You appear to manage your
                  fixed costs predictably, suggesting a stable budgeting habit.
                  However, there are also moderate but noticeable spikes in
                  discretionary areas such as dining out, entertainment, and
                  online shopping—often clustered toward weekends or just after
                  payday—which may indicate moments of reward spending or social
                  activity.
                  <br />
                  <br />
                  Looking at the month as a whole, your spending balance seems
                  healthy, but there&apos;s room to fine-tune allocation toward
                  savings or investment without reducing quality of life. Small
                  adjustments—like consolidating food spending or reviewing
                  subscription renewals—could help trim less intentional
                  expenses. Overall, your spending reflects a controlled but
                  flexible approach, with opportunities to redirect surplus cash
                  toward long-term goals.
                </p>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card className="flex flex-col w-full items-start bg-[#f8f8f8] rounded-[20px] overflow-hidden border-0">
        <CardContent className="flex flex-col w-full p-0">
          <header className="h-10 flex items-start gap-16 px-[42px] py-0 bg-[#f8f8f8] w-full">
            <div className="inline-flex items-center px-2 py-0 self-stretch">
              <span className="font-[number:var(--small-font-weight)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] font-small [font-style:var(--small-font-style)] text-[length:var(--small-font-size)] text-[#050101]">
                DATE
              </span>
            </div>

            <div className="flex w-[366px] items-center px-2 py-0 self-stretch">
              <span className="font-[number:var(--small-font-weight)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] font-small [font-style:var(--small-font-style)] text-[length:var(--small-font-size)] text-[#050101]">
                PAYEE
              </span>
            </div>

            <div className="flex w-[124px] items-center px-2 py-0 self-stretch">
              <span className="font-[number:var(--small-font-weight)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] font-small [font-style:var(--small-font-style)] text-[length:var(--small-font-size)] text-[#050101]">
                VALUE
              </span>
            </div>

            <div className="flex w-[124px] items-center px-2 py-0 self-stretch">
              <span className="font-[number:var(--small-font-weight)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] font-small [font-style:var(--small-font-style)] text-[length:var(--small-font-size)] text-[#050101]">
                CATEGORY
              </span>
            </div>
          </header>

          <div className="flex flex-col items-center gap-2 px-0 py-2 w-full">
            {transactionData.map((transaction, index) => (
              <div
                key={`transaction-${index}`}
                className="flex flex-col items-start justify-center gap-2 px-4 py-0 w-full"
              >
                <div className="inline-flex h-6 items-center gap-2">
                  <Checkbox className="w-4 h-4 bg-white rounded border border-solid border-[#757575]" />

                  <div className="flex w-[109px] items-center px-2 py-0 self-stretch">
                    <span className="mt-[-1.00px] font-small font-[number:var(--small-font-weight)] text-[#020202] text-[length:var(--small-font-size)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] whitespace-nowrap [font-style:var(--small-font-style)]">
                      {transaction.date}
                    </span>
                  </div>

                  <div className="flex w-[419px] items-center px-2 py-0 self-stretch">
                    <span className="font-body font-[number:var(--body-font-weight)] text-[#020202] text-[length:var(--body-font-size)] tracking-[var(--body-letter-spacing)] leading-[var(--body-line-height)] whitespace-nowrap [font-style:var(--body-font-style)]">
                      {transaction.payee}
                    </span>
                  </div>

                  <div className="flex w-[178px] items-center px-2 py-0 self-stretch">
                    <span className="font-body font-[number:var(--body-font-weight)] text-[#020202] text-[length:var(--body-font-size)] tracking-[var(--body-letter-spacing)] leading-[var(--body-line-height)] whitespace-nowrap [font-style:var(--body-font-style)]">
                      {transaction.value}
                    </span>
                  </div>

                  <div className="flex w-[196px] items-center px-2 py-0 self-stretch">
                    <Badge
                      className={`flex w-[188px] items-center justify-center gap-2 self-stretch mr-[-8.00px] ${transaction.categoryColor} rounded-full border-0 h-auto`}
                    >
                      <span className="mt-[-1.00px] font-small font-[number:var(--small-font-weight)] text-[#020202] text-[length:var(--small-font-size)] tracking-[var(--small-letter-spacing)] leading-[var(--small-line-height)] whitespace-nowrap [font-style:var(--small-font-style)]">
                        {transaction.category}
                      </span>
                    </Badge>
                  </div>
                </div>

                <Separator className="w-[1036px] h-px mr-[-16.00px]" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </section>
  );
};
