"""检查数据完整性"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.milvus import get_collection_stats

# 检查向量数据库
stats = get_collection_stats()
print("=" * 50)
print("向量数据库统计")
print("=" * 50)
print(f"集合名称: {stats['collection_name']}")
print(f"记录数量: {stats['row_count']}")

# 检查data目录下的文件
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cook", "dishes")
if os.path.exists(data_dir):
    md_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root, f))

    print("\n" + "=" * 50)
    print("文档统计")
    print("=" * 50)
    print(f"Markdown文件数量: {len(md_files)}")

    # 统计每个目录下的文件
    print("\n按目录分类:")
    dir_count = {}
    for f in md_files:
        rel_path = os.path.relpath(f, data_dir)
        dir_name = os.path.dirname(rel_path)
        if dir_name not in dir_count:
            dir_count[dir_name] = 0
        dir_count[dir_name] += 1

    for dir_name, count in sorted(dir_count.items()):
        if dir_name:
            print(f"  {dir_name}: {count} 个文件")
else:
    print(f"\ndata目录不存在: {data_dir}")

# 计算预期的chunk数量
# 每个菜谱大约有: 1 ingredient + N steps + 1 tip + M images
# 平均约 10-15 chunks per recipe
print("\n" + "=" * 50)
print("数据完整性检查")
print("=" * 50)

expected_min_chunks = len(md_files) * 5  # 最少每个菜谱5个chunks
expected_max_chunks = len(md_files) * 20  # 最多每个菜谱20个chunks

actual_chunks = stats['row_count']

if actual_chunks >= expected_min_chunks:
    print(f"[OK] 向量数据完整 ({actual_chunks} chunks >= {expected_min_chunks} 最小预期)")
else:
    print(f"[WARNING] 向量数据可能不完整 ({actual_chunks} chunks < {expected_min_chunks} 最小预期)")

print(f"\n结论: {len(md_files)} 个Markdown文件 -> {actual_chunks} 个向量chunks")
