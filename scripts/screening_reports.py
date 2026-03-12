from __future__ import annotations

import pandas as pd


def build_rankings(df: pd.DataFrame, cns_threshold: float, ad_threshold: float | None, topk: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    ranked_all = df.sort_values("score_mean", ascending=False).reset_index(drop=True)
    ranked_cns = ranked_all[ranked_all.get("cns_mpo", 0) >= cns_threshold].copy() if "cns_mpo" in ranked_all.columns else ranked_all.iloc[0:0].copy()
    ranked_ad = ranked_all[ranked_all.get("ad_distance", 1e9) <= ad_threshold].copy() if ad_threshold is not None and "ad_distance" in ranked_all.columns else ranked_all.copy()
    ranked_both = ranked_ad[ranked_ad.get("cns_mpo", 0) >= cns_threshold].copy() if "cns_mpo" in ranked_ad.columns else ranked_ad

    best = ranked_both if len(ranked_both) else ranked_ad if len(ranked_ad) else ranked_cns if len(ranked_cns) else ranked_all
    sel = pd.DataFrame(
        [
            {"list": "all", "count": len(ranked_all)},
            {"list": "cns_like", "count": len(ranked_cns)},
            {"list": "in_domain", "count": len(ranked_ad)},
            {"list": "cns_like_in_domain", "count": len(ranked_both)},
            {"list": "topk_export", "count": min(topk, len(best))},
        ]
    )
    return {
        "ranked_all": ranked_all,
        "ranked_cns_like": ranked_cns,
        "ranked_in_domain": ranked_ad,
        "ranked_cns_like_in_domain": ranked_both,
        "best": best,
    }, sel
