import json
import sys

filename = "/Users/roop/Library/CloudStorage/OneDrive-UniversityofFlorida/Courses/Spring 26 EGN6933/Code/CineMatch/src/ui/gradio_app.ipynb"
try:
    with open(filename, "r") as f:
        nb = json.load(f)
except Exception as e:
    print("Failed to read", e)
    sys.exit(1)

for c in nb["cells"]:
    if c.get("cell_type") == "code":
        source = c["source"]
        for i, line in enumerate(source):
            if "mongo_get_collection().find_one" in line:
                source[i] = line.replace("mongo_get_collection()", "mongo_db['users']")
                print("Patched mongo_get_collection")
            if "        'id': int(record.get('id', record.get('tmdb_id', 0)))," in line:
                source[i] = "        'id': int(record.get('id') or record.get('tmdb_id') or 0),\n"
                print("Patched safe ID")
            if "        'title': record.get('title', record.get('Title', GRADIO_TITLE_BY_ID.get(int(record.get('id', record.get('tmdb_id', 0))), '')))," in line:
                source[i] = "        'title': record.get('title', record.get('Title', GRADIO_TITLE_BY_ID.get(int(record.get('id') or record.get('tmdb_id') or 0), ''))),\n"
                print("Patched safe Title")
            if "        'poster_path': record.get('poster_path', GRADIO_POSTER_BY_ID.get(int(record.get('id', record.get('tmdb_id', 0))), ''))," in line:
                source[i] = "        'poster_path': record.get('poster_path', GRADIO_POSTER_BY_ID.get(int(record.get('id') or record.get('tmdb_id') or 0), '')),\n"
                print("Patched safe Poster")

with open(filename, "w") as f:
    json.dump(nb, f, indent=2)

print("All patches applied!")
