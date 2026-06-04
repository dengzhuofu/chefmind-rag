"""测试解析器是否正确提取用量信息"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.recipe_parser import recipe_parser

# 测试解析清蒸鲈鱼
recipe_path = r"C:\Users\Eason\Desktop\rag-langchain\data\cook\dishes\aquatic\清蒸鲈鱼\清蒸鲈鱼.md"
recipe = recipe_parser.parse_file(recipe_path)

print("=" * 50)
print(f"菜名: {recipe.title}")
print("=" * 50)

print("\n食材列表:")
for i, ing in enumerate(recipe.ingredients, 1):
    print(f"  {i}. {ing}")

print("\n步骤:")
for i, step in enumerate(recipe.steps, 1):
    print(f"  {i}. {step[:60]}...")

# 测试解析另一个菜谱
print("\n" + "=" * 50)
recipe_path2 = r"C:\Users\Eason\Desktop\rag-langchain\data\cook\dishes\meat_dish\宫保鸡丁\宫保鸡丁.md"
if os.path.exists(recipe_path2):
    recipe2 = recipe_parser.parse_file(recipe_path2)
    print(f"菜名: {recipe2.title}")
    print("=" * 50)
    print("\n食材列表:")
    for i, ing in enumerate(recipe2.ingredients, 1):
        print(f"  {i}. {ing}")
