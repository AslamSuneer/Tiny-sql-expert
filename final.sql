SELECT u.name, u.email FROM users u JOIN orders o ON u.user_id = o.user_id JOIN products p ON p.product_id = o.product_id WHERE p.category = 'Electronics' COLLATE NOCASE;
