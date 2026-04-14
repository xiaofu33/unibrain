import psycopg
import sys
import os

# Ensure current dir is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import settings

def test_connection():
    print("========================================")
    print("Testing PostgreSQL Connection...")
    host = settings.POSTGRES_HOST
    port = settings.POSTGRES_PORT
    user = settings.POSTGRES_USER
    password = settings.POSTGRES_PASSWORD
    dbname = settings.POSTGRES_DB
    
    print(f"Host: {host}:{port}")
    print(f"User: {user}")
    print(f"Target DB: {dbname}")
    print("========================================\n")
    
    try:
        print("[1] 正在尝试以超级用户的身份连接默认的 PostgreSQL...")
        # Connecting to 'postgres' to potentially create 'unibrain_db'
        # Default installing by brew doesn't have a password usually for the local user
        conn_str_baseline = f"host={host} port={port} user={user} password={password} dbname=postgres"
        conn = psycopg.connect(conn_str_baseline, autocommit=True)
        print("✅ 成功连接到首选实例 (postgres).")
        
        cur = conn.cursor()
        print(f"\n[2] 检查专属知识库数据库 '{dbname}' 是否存在...")
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        if not cur.fetchone():
            print(f"⚠️ 未找到数据库。正在为您创建 '{dbname}'...")
            cur.execute(f'CREATE DATABASE "{dbname}"')
            print(f"✅ 成功创建专属数据库 '{dbname}'.")
        else:
            print(f"✅ 数据库 '{dbname}' 已经存在.")
        conn.close()
        
        print("\n[3] 正在进入专属数据库并全量配置 pgvector 插件...")
        conn_str_target = f"host={host} port={port} user={user} password={password} dbname={dbname}"
        conn = psycopg.connect(conn_str_target, autocommit=True)
        print(f"✅ 成功连接到目标数据库 '{dbname}'.")
        
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✅ 成功在当前数据库环境启用了 pgvector 向量搜索插件.")
        
        cur.execute("SELECT typname FROM pg_type WHERE typname = 'vector'")
        if cur.fetchone():
            print("✅ 校验发现 'vector' 原生数据类型可用.")
        else:
            print("❌ 警告：启用了拓展模块，但找不到 'vector' 数据类型。")
            
        print("\n🎉 === 所有的 Postgres 数据库测试均圆满通过 (ALL TESTS PASSED) === 🎉")
        conn.close()
    except Exception as e:
        print("\n❌ !!! 数据库连接测试失败 (ERROR DETECTED) !!! ❌")
        print("详细错误日志如下：")
        print(e)
        print("\n请检查：")
        print("1. 您的 brew PostgreSQL 是否正在运行。")
        print("2. pg_hba.conf 以及密码是否配置正确（默认 Brew 下可能是当前的 Mac 用户名且没有密码）。")

if __name__ == "__main__":
    test_connection()
