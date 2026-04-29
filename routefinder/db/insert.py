def insert_area(cur, area, parent_id=None):
    """Recursively insert an area/route tree into the database."""
    if not area or (not area.get("url") and not area.get("subareas") and not area.get("routes")):
        return

    if area.get("type") == "sitemap":
        for sub in area.get("subareas", {}).values():
            insert_area(cur, sub, parent_id)
        return

    area_url = area.get("url")
    area_name = area_url.split("/")[-1]

    cur.execute("""
        INSERT INTO areas (name, parent_id, url)
        VALUES (%s, %s, %s)
        ON CONFLICT (url) DO NOTHING
        RETURNING id
    """, (area_name, parent_id, area_url))
    row = cur.fetchone()

    if row:
        area_id = row[0]
    else:
        cur.execute("SELECT id FROM areas WHERE url=%s", (area_url,))
        area_id = cur.fetchone()[0]

    for sub in area.get("subareas", {}).values():
        if sub and sub.get("url"):
            insert_area(cur, sub, area_id)

    for route in area.get("routes", []):
        if not route or not route.get("url"):
            continue

        cur.execute("""
            INSERT INTO routes (area_id, name, grade, url, type, description, location, gps)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
            RETURNING id
        """, (
            area_id,
            route.get("name"),
            route.get("grade"),
            route.get("url"),
            route.get("Type:", None),
            route.get("Description", None),
            route.get("Location", None),
            route.get("GPS:", None),
        ))
        row = cur.fetchone()
        if row:
            route_id = row[0]
        else:
            cur.execute("SELECT id FROM routes WHERE url=%s", (route.get("url"),))
            route_id = cur.fetchone()[0]

        for img_url in route.get("images", []):
            cur.execute("""
                INSERT INTO route_images (route_id, image_url)
                VALUES (%s, %s)
                ON CONFLICT (image_url) DO NOTHING
            """, (route_id, img_url))


def build_test_tree(area, max_subareas=2, max_routes=2):
    """Recursively trim a tree for quick insertion tests."""
    if not area:
        return None
    test_area = {
        "type": area.get("type"),
        "name": area.get("name"),
        "url": area.get("url"),
        "subareas": {},
        "routes": area.get("routes", [])[:max_routes],
    }
    for k, sub in list(area.get("subareas", {}).items())[:max_subareas]:
        sub_test = build_test_tree(sub, max_subareas, max_routes)
        if sub_test:
            test_area["subareas"][k] = sub_test
    return test_area


def count_tree(tree):
    areas = routes = images = 0

    def _count(area):
        nonlocal areas, routes, images
        if not area:
            return
        areas += 1
        routes += len(area.get("routes", []))
        for r in area.get("routes", []):
            images += len(r.get("images", []))
        for sub in area.get("subareas", {}).values():
            _count(sub)

    _count(tree)
    return areas, routes, images
