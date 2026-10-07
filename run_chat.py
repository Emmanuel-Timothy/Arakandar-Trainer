"""Chat with the local Arakandar market research model."""
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

from app.agent.llm.local_model import TransformersLocalModel
from app.agent.orchestrator import ArakandarAgent
from app.agent.storage import AgentStore
from app.data.market import MarketIngestor, YFinanceProvider
from app.data.news import google_news_feed


def _feeds(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        ticker, separator, url = value.partition("=")
        if not separator or not ticker or not url:
            raise ValueError("--rss-feed must use TICKER=URL")
        result[ticker] = url
    return result


def _symbol_candidates() -> set[str]:
    candidates = {
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "BYAN.JK", "GOTO.JK", "AMMN.JK",
        "BBNI.JK", "BRIS.JK", "ASII.JK", "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "GOOGL", "AVGO", "AMD", "PLTR", "ORCL", "CRM", "NFLX", "COST", "WMT",
        "UNH", "LLY", "JPM", "V", "MA", "XOM", "CVX", "SAP.DE", "SIE.DE",
        "ASML.AS", "AI.PA", "OR.PA", "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L",
        "0700.HK", "9988.HK", "1810.HK", "1299.HK", "2318.HK", "600519.SS",
        "601318.SS", "601857.SS", "601888.SS", "D05.SI", "O39.SI", "U11.SI",
        "7203.T", "6758.T", "8306.T", "6501.T", "8035.T", "8058.T", "9432.T",
        "2914.T", "7267.T",
    }
    watchlist_path = Path(__file__).resolve().parent / "watchlists" / "ihsg.txt"
    try:
        with watchlist_path.open("r", encoding="utf-8") as handle:
            candidates.update(
                line.strip().upper()
                for line in handle
                if line.strip() and not line.strip().startswith("#")
            )
    except OSError:
        pass
    return {symbol.upper() for symbol in candidates}


def _indonesia_watchlist() -> list[str]:
    """Return all Indonesian symbols configured in the IHSG watchlist."""
    return sorted(symbol for symbol in _symbol_candidates() if symbol.endswith(".JK"))


def _global_watchlist() -> list[str]:
    """Return the complete configured multi-market universe."""
    return sorted(_symbol_candidates())


def _market_watchlist(suffixes: tuple[str, ...]) -> list[str]:
    if suffixes == ("",):
        return sorted(symbol for symbol in _global_watchlist() if "." not in symbol)
    return sorted(symbol for symbol in _global_watchlist() if symbol.endswith(suffixes))


def _market_label(question: str) -> str:
    normalized = question.upper()
    if "NASDAQ" in normalized:
        return "NASDAQ/US"
    if "S&P" in normalized or "SP500" in normalized or "S&P 500" in normalized:
        return "S&P 500 representative US"
    if "NIKKEI" in normalized or "JAPAN" in normalized:
        return "Japan/Nikkei representative"
    if "CHINA" in normalized or "HONG KONG" in normalized or "SHANGHAI" in normalized:
        return "China/Hong Kong"
    if "SINGAPORE" in normalized:
        return "Singapore"
    if "EUROPE" in normalized or "EUROPEAN" in normalized:
        return "Europe"
    if "IHSG" in normalized or "INDONESIA" in normalized:
        return "IHSG/Indonesia"
    return "global"


def _extract_tickers(question: str, store: AgentStore, current: str) -> list[str]:
    """Extract one or many tickers from a question, including compare/watchlist-style prompts."""
    try:
        import sqlite3
        con = sqlite3.connect(str(store.path))
        rows = con.execute("SELECT DISTINCT ticker FROM market_bars").fetchall()
        con.close()
        available = {row[0].upper() for row in rows}
    except Exception:
        available = set()
    available.update(_symbol_candidates())

    normalized = question.upper().replace("-", ".")
    found: list[str] = []
    for ticker in sorted(available, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", normalized):
            found.append(ticker)
        short = ticker.removesuffix(".JK")
        if short != ticker and re.search(rf"(?<![A-Z0-9]){re.escape(short)}(?![A-Z0-9])", normalized):
            found.append(ticker)

    unique: list[str] = []
    for ticker in found:
        if ticker not in unique:
            unique.append(ticker)
    if unique:
        return unique

    market_groups = (
        (("NASDAQ", "S&P", "SP500", "NYSE", "US STOCK", "AMERICAN"), ("",), "US"),
        (("IHSG", "INDONESIA", "INDONESIAN"), (".JK",), "Indonesia"),
        (("NIKKEI", "JAPAN", "JAPANESE"), (".T",), "Japan"),
        (("CHINA", "HONG KONG", "SHANGHAI", "CHINESE"), (".HK", ".SS", ".SZ"), "China/Hong Kong"),
        (("SINGAPORE", "STRAITS TIMES"), (".SI",), "Singapore"),
        (("EUROPE", "EUROPEAN", "DAX", "CAC", "FTSE"), (".DE", ".PA", ".AS", ".L", ".SW"), "Europe"),
    )
    for keywords, suffixes, _ in market_groups:
        if any(keyword in normalized for keyword in keywords):
            return [symbol for symbol in _market_watchlist(suffixes) if symbol in available]

    global_market_keywords = [
        "ALL MARKET", "ALL MARKETS", "GLOBAL MARKET", "GLOBAL MARKETS",
        "WORLD MARKET", "WORLDWIDE", "INTERNATIONAL STOCK", "GLOBAL STOCK",
        "BEST STOCK WORLDWIDE", "BEST STOCK GLOBALLY", "ACROSS MARKETS",
    ]
    if any(keyword in normalized for keyword in global_market_keywords):
        return [symbol for symbol in _global_watchlist() if symbol in available]

    general_market_keywords = [
        "BEST TO BUY", "BEST BUY", "GOOD BUY", "WHAT SHOULD I BUY", "WHAT TO BUY",
        "BUY TOMORROW", "FOR TOMORROW", "BEST STOCK", "GOOD STOCK", "TOP STOCK",
        "STOCKS TO BUY", "STOCK TO BUY", "INVEST IN", "BUY IN IHSG", "IHSG",
        "MARKET OUTLOOK", "MARKET VIEW", "MARKET ANALYSIS", "MARKET UPDATE",
        "STOCK MARKET", "INDONESIAN STOCK", "INDONESIA STOCK", "WHICH STOCK",
        "STOCK RECOMMENDATION", "INVESTMENT IDEA", "OPPORTUNITY", "PORTFOLIO",
    ]
    if any(keyword in normalized for keyword in general_market_keywords):
        return [symbol for symbol in _indonesia_watchlist() if symbol in available]

    for brand, symbol in {"GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "APPLE": "AAPL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA", "AMAZON": "AMZN", "META": "META", "FACEBOOK": "META"}.items():
        if brand in normalized and "STOCK" in normalized:
            return [symbol]

    # Questions without a ticker but with market intent should scan the default
    # universe instead of silently being answered for the startup ticker.
    if any(word in normalized for word in ("STOCK", "MARKET", "IHSG", "BUY", "INVEST", "SHARE")):
        return [symbol for symbol in _indonesia_watchlist() if symbol in available]

    return [current]


def _ticker_from_question(question: str, store: AgentStore, current: str) -> str:
    """Choose the primary ticker from a question, preferring the first explicit mention."""
    tickers = _extract_tickers(question, store, current)
    return tickers[0] if tickers else current


def _available_tickers(store: AgentStore) -> list[str]:
    """Return known symbols for the interactive command menu."""
    try:
        import sqlite3
        with sqlite3.connect(str(store.path)) as connection:
            rows = connection.execute("SELECT DISTINCT ticker FROM market_bars").fetchall()
        stored = {str(row[0]).upper() for row in rows if row[0]}
    except Exception:
        stored = set()
    return sorted(stored | _symbol_candidates())


def _print_help() -> None:
    print(
        "\nCommands:\n"
        "  /help                 Show this command list\n"
        "  /listticker           List supported and previously cached tickers\n"
        "  /ticker SYMBOL        Change the active ticker\n"
        "  /quit or /exit        Leave Arakandar\n"
        "\nYou can also ask questions such as: Compare BBCA and BBRI\n"
    )


def _default_model_path() -> Path:
    """Prefer a trained adapter, then 3B, while retaining a smaller fallback."""
    models_root = Path(__file__).resolve().parent / "models" / "base"
    adapter = Path(__file__).resolve().parent / "models" / "adapters" / "arakandar-3b-lora"
    preferred = models_root / "arakandar-3b"
    fallback = models_root / "arakandar-1.5b"
    if (adapter / "adapter_config.json").is_file():
        return adapter
    return preferred if preferred.is_dir() else fallback


def _latest_close(store: AgentStore, ticker: str) -> tuple[str, str | None]:
    try:
        bars = store.latest_market_bars(ticker, limit=1)
        if not bars:
            return ("n/a", None)
        bar = bars[-1]
        return (f"{float(bar.get('close', 0.0)):.2f}", bar.get("timestamp"))
    except Exception:
        return ("n/a", None)


def _comparison_table(rows: list[dict[str, str]]) -> str:
    headers = ["Ticker", "Signal", "Close", "RSI", "MACD", "Sentiment", "Trend", "Confidence"]
    formatted = []
    for row in rows:
        confidence = str(row.get("confidence", "n/a"))
        if confidence != "n/a":
            try:
                confidence = f"{float(confidence):.0%}"
            except (TypeError, ValueError):
                confidence = str(confidence)
        formatted.append({
            "Ticker": row["ticker"],
            "Signal": row["signal"],
            "Close": row["close"],
            "RSI": row.get("rsi", "n/a"),
            "MACD": row.get("macd", "n/a"),
            "Sentiment": row.get("sentiment", "n/a"),
            "Trend": row.get("trend", "n/a"),
            "Confidence": confidence,
        })

    width_rows = [*formatted, {key: header for key, header in zip(headers, headers)}]
    widths = {
        key: max(len(str(item.get(key, ""))) for item in width_rows)
        for key in headers
    }
    head_line = " | ".join(str(value).ljust(widths[key]) for key, value in zip(headers, headers))
    divider = "-+-".join("-" * widths[key] for key in headers)
    lines = [head_line, divider]
    for row in formatted:
        lines.append(" | ".join(str(row.get(key, "")).ljust(widths[key]) for key in headers))
    return "\n".join(lines)


def _comparison_reason(answer: str, signal: str) -> str:
    """Make model prose safe for one-line terminal ranking output."""
    compact = " ".join(answer.split())
    if not compact:
        return f"Signal is {signal}."
    return textwrap.shorten(compact, width=60, placeholder="...")


def _evidence_reason(response: dict) -> str:
    """Build a complete short ranking reason from deterministic evidence."""
    evidence = response.get("evidence") if isinstance(response.get("evidence"), dict) else {}
    signal = response.get("signal", "HOLD")
    rsi = evidence.get("rsi14")
    macd = evidence.get("macd_diff")
    confidence = evidence.get("confidence")
    parts = [f"{signal} signal"]
    if rsi is not None:
        rsi_view = "oversold" if float(rsi) <= 30 else "overbought" if float(rsi) >= 70 else "neutral RSI"
        parts.append(f"RSI {float(rsi):.1f} ({rsi_view})")
    if macd is not None:
        momentum = "positive" if float(macd) > 0 else "negative" if float(macd) < 0 else "flat"
        parts.append(f"MACD {float(macd):.2f} ({momentum} momentum)")
    if confidence is not None:
        parts.append(f"confidence {float(confidence):.0%}")
    return "; ".join(parts) + "."


def main() -> int:
    default_model = _default_model_path()
    parser = argparse.ArgumentParser(description="Chat with local Arakandar market intelligence")
    parser.add_argument("--ticker", action="append", default=["BBCA.JK"], help="Ticker(s) to seed the chat with; repeat to add more")
    parser.add_argument("--period", default="2y", help="Historical market lookback to fetch for each ticker (for example 1y, 2y, 5y)")
    parser.add_argument("--interval", default="1d", help="Bar interval for market data, for example 1d or 1h")
    parser.add_argument(
        "--llm-model",
        default=str(default_model) if default_model.is_dir() else None,
        help="Local Transformers model path (use --heuristic to disable)",
    )
    parser.add_argument("--heuristic", action="store_true", help="Use the deterministic summary instead of the local LLM")
    parser.add_argument("--device", default="auto", choices=("cuda", "cpu", "auto"))
    parser.add_argument("--question", help="Ask one question and exit (non-interactive)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached/local data only; live market and news fetching is enabled by default",
    )
    parser.add_argument("--refresh", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rss-feed", action="append", default=[], metavar="TICKER=URL")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--state", default="data/agent_state.sqlite3")
    parser.add_argument("--signal-model", default="models/champion_lgbm.pkl")
    args = parser.parse_args()

    try:
        rss_feeds = _feeds(args.rss_feed)
    except ValueError as exc:
        parser.error(str(exc))

    live = not args.offline
    requested_tickers = list(dict.fromkeys(args.ticker or ["BBCA.JK"]))
    primary_ticker = requested_tickers[0]
    store = AgentStore(Path(args.state))
    if live:
        configured_tickers = set(requested_tickers) | set(rss_feeds)
        rss_feeds = {
            ticker: rss_feeds.get(ticker, google_news_feed(ticker))
            for ticker in configured_tickers
        }
    language_model = (
        None
        if args.heuristic or not args.llm_model
        else TransformersLocalModel(
            args.llm_model,
            device=args.device,
            model_name=f"Arakandar-{Path(args.llm_model).name}",
        )
    )
    agent = ArakandarAgent(
        Path(args.data_dir),
        Path(args.signal_model),
        store,
        language_model=language_model,
        rss_feeds=rss_feeds,
    )
    ingestor = MarketIngestor(YFinanceProvider(), store) if live else None

    history: list[dict[str, str]] = []
    ticker = primary_ticker
    print("\n" + "="*55)
    print("  Arakandar Market Intelligence")
    print("="*55)
    if len(requested_tickers) == 1:
        print("  Mode        : General market questions scan Indonesia")
    else:
        print(f"  Scan tickers : {', '.join(requested_tickers)}")
    print(f"  Data     : {'live market + news (refreshed per question)' if live else 'cached data only'}")
    print(f"  Lookback : {args.period} {args.interval}")
    print("="*55)
    print("  Try: /ticker AAPL | Compare BBCA and BBRI | Best buy in NASDAQ")
    print("="*55 + "\n")

    def answer(question: str) -> None:
        nonlocal history, ticker
        detected = _extract_tickers(question, store, ticker)
        if len(detected) > 1:
            observations: list[tuple[str, dict]] = []
            # A broad scan ranks deterministic evidence; loading and generating
            # with the 3B model once per ticker adds latency without improving
            # the signal ranking. Specific ticker questions still use the LLM.
            language_model = agent.language_model
            agent.language_model = None
            print(f"\nScanning {len(detected)} tickers using fast deterministic signals...")
            for symbol in detected:
                try:
                    if live:
                        agent.news_ingestor.provider.feeds.setdefault(symbol, google_news_feed(symbol))
                    if ingestor:
                        ingestor.sync(symbol, period=args.period, interval=args.interval)
                    response = agent.chat(question, symbol, history=history)
                    close_value, _ = _latest_close(store, symbol)
                    evidence = response["evidence"] if isinstance(response.get("evidence"), dict) else {}
                    rsi = evidence.get("rsi14")
                    macd = evidence.get("macd_diff")
                    sentiment = evidence.get("news_sentiment", "n/a")
                    trend = "UP" if (rsi is not None and float(rsi) < 70) else "DOWN" if (rsi is not None and float(rsi) > 70) else "SIDE"
                    response_row = {
                        "ticker": symbol,
                        "signal": response["signal"],
                        "close": close_value,
                        "rsi": f"{float(rsi):.1f}" if rsi is not None else "n/a",
                        "macd": f"{float(macd):.2f}" if macd is not None else "n/a",
                        "sentiment": f"{sentiment}",
                        "trend": trend,
                        "confidence": evidence.get("confidence", "n/a"),
                    }
                    observations.append((symbol, response, response_row))
                    if len(observations) % 5 == 0 or len(observations) == len(detected):
                        print(f"  Progress: {len(observations)}/{len(detected)}")
                except Exception as exc:
                    print(f"Skipping {symbol}: {exc}")
            agent.language_model = language_model

            if not observations:
                print("No ticker data was available for this scan.")
                return

            signal_rank = {"BUY": 3, "HOLD": 2, "SELL": 1}
            ranked = sorted(observations, key=lambda item: signal_rank.get(item[1]["signal"], 0), reverse=True)
            top = ranked[0][0]
            if not any(symbol in question.upper() for symbol in detected):
                print(f"\nGeneral {_market_label(question)} market scan: comparing {len(detected)} symbols.")
            print("\nComparison view:\n")
            print(_comparison_table([row for _, _, row in observations]))
            print("\nBest buy ranking:")
            ordering = 1
            for symbol, response, _ in ranked:
                reason = _evidence_reason(response)
                print(f"{ordering}. {symbol} — {response['signal']} — {reason}")
                ordering += 1
            print(f"\nBest fit from the set: {top}")
            return

        detected_ticker = _ticker_from_question(question, store, ticker)
        if detected_ticker != ticker:
            ticker = detected_ticker
            history = []
        if live:
            agent.news_ingestor.provider.feeds.setdefault(ticker, google_news_feed(ticker))
        if ingestor:
            ingestor.sync(ticker, period=args.period, interval=args.interval)
        response = agent.chat(question, ticker, history=history)
        print(f"\n{ticker} | {response['signal']}\n{response['answer']}\n")
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response["answer"]},
            ]
        )

    if args.question:
        answer(args.question)
        return 0

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        command = question.lower()
        if command in {"/quit", "/exit"}:
            return 0
        if command == "/help":
            _print_help()
            continue
        if command == "/listticker":
            print("\nAvailable tickers:\n" + "\n".join(f"- {symbol}" for symbol in _available_tickers(store)) + "\n")
            continue
        if command.startswith("/ticker "):
            requested = question.split(None, 1)[1].strip().upper()
            if not requested:
                print("Usage: /ticker SYMBOL")
                continue
            ticker = requested
            history = []
            print(f"Ticker changed to {ticker}")
            continue
        if command.startswith("/"):
            print("Unknown command. Type /help for available commands.")
            continue
        try:
            answer(question)
        except Exception as exc:
            print(f"Arakandar error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
