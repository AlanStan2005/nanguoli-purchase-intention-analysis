def strategy_from_factor(factor: str) -> str:
    mapping = {
        "quality_score": "强化质量追溯与分级包装",
        "brand_awareness": "统一地标品牌视觉和产地认证表达",
        "channel_trust": "优化电商客服、售后和复购触达",
        "price_acceptance": "设置不同规格价格带和组合装",
    }
    return mapping.get(factor, "纳入后续消费者访谈验证")
