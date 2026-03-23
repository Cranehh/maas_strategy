"""
One-time utility: Extract parameters from Biogeme .pickle or .html files.
Run this script to verify/update CH3_PARAMS in config.py.
"""
import re
import sys
import os


def extract_from_html(html_path):
    """Extract beta values from Biogeme HTML report."""
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    params = {}
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL)
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) >= 2:
            name = cells[0].strip()
            if re.match(r'^[A-Za-z_]', name) and '<' not in name:
                try:
                    val = float(cells[1].strip())
                    params[name] = val
                except ValueError:
                    pass
    return params


def extract_from_pickle(pickle_path):
    """Extract beta values from Biogeme pickle results."""
    try:
        import biogeme.results as res
        results = res.bioResults(pickleFile=pickle_path)
        return results.getBetaValues()
    except ImportError:
        print("biogeme not installed, falling back to HTML extraction")
        return None
    except Exception as e:
        print(f"Pickle loading failed: {e}")
        return None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(
        base_dir, '已有代码', 'MaaS 附加值相关',
        'HCM~nofactor5 去掉套餐中的ebiike.html'
    )
    pickle_path = os.path.join(
        base_dir, '已有代码', 'MaaS 附加值相关',
        'HCM~nofactor5 去掉套餐中的ebiike.pickle'
    )

    # Try pickle first, fallback to HTML
    params = extract_from_pickle(pickle_path)
    if params is None:
        params = extract_from_html(html_path)

    print(f"Extracted {len(params)} parameters:\n")
    print("CH3_PARAMS = {")
    for k, v in sorted(params.items()):
        print(f"    '{k}': {v},")
    print("}")


if __name__ == '__main__':
    main()
