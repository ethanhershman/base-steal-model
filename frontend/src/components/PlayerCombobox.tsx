import { useQuery } from "@tanstack/react-query"
import { Check, ChevronsUpDown, Loader2 } from "lucide-react"
import { useEffect, useState } from "react"

import { searchPlayers } from "@/lib/api"
import type { PlayerRole, PlayerSearchResult } from "@/lib/types"
import { cn } from "@/lib/utils"

import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])
  return debounced
}

interface PlayerComboboxProps {
  role: PlayerRole
  label: string
  selected: PlayerSearchResult | null
  onSelect: (player: PlayerSearchResult) => void
}

// Reused 3x (runner/pitcher/catcher) -- a shadcn Command+Popover combobox
// backed by /api/players/search. Selecting a player auto-fills their real
// stats into the parent form (see App.tsx); no manual override UI, per the
// product decision to search real players rather than hand-enter stats.
export function PlayerCombobox({ role, label, selected, onSelect }: PlayerComboboxProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const debouncedQuery = useDebouncedValue(query, 200)
  const searchReady = debouncedQuery.trim().length >= 2

  const { data: results, isFetching } = useQuery({
    queryKey: ["players", role, debouncedQuery],
    queryFn: () => searchPlayers(role, debouncedQuery),
    enabled: searchReady,
  })

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between font-normal"
        >
          {selected ? selected.name : `Search ${label.toLowerCase()}...`}
          <ChevronsUpDown className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0">
        <Command shouldFilter={false}>
          <CommandInput
            placeholder={`Type a ${label.toLowerCase()} name...`}
            value={query}
            onValueChange={setQuery}
          />
          <CommandList>
            {isFetching && (
              <div className="flex items-center justify-center py-4">
                <Loader2 className="size-4 animate-spin opacity-50" />
              </div>
            )}
            {!isFetching && searchReady && (
              <CommandEmpty>No {label.toLowerCase()} found.</CommandEmpty>
            )}
            <CommandGroup>
              {results?.map((player) => (
                <CommandItem
                  key={player.id}
                  value={player.id}
                  onSelect={() => {
                    onSelect(player)
                    setOpen(false)
                    setQuery("")
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 size-4",
                      selected?.id === player.id ? "opacity-100" : "opacity-0",
                    )}
                  />
                  {player.name}
                  {player.team && (
                    <span className="ml-auto text-xs text-muted-foreground">{player.team}</span>
                  )}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
