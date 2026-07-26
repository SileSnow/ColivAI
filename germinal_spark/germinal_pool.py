"""
GerminalPool — 干细胞池
GermCell — 单个干细胞（不对称分裂单元）
RadialFiber — 放射纤维（干细胞→子代梯度通道）

不参与前向计算，只负责 spark() 生成新柱
"""

import torch
import torch.nn as nn
from typing import List
from spark_column import SparkColumn, Neuroblast


class RadialFiber(nn.Module):
    """
    放射纤维：连接干细胞和子代的梯度通道
    
    在生物学中，放射状胶质细胞伸出一根长长的突起（放射纤维）
    直达皮层表面。新生的神经母细胞沿着这根纤维向上迁移。
    
    在工程中，RadialFiber 定义了干细胞参数如何"投影"到子代柱的初始参数。
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 简单的线性投影：干细胞参数空间 → 子代参数空间
        self.fiber_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def guide(self, stem_params: dict, target_column: SparkColumn, noise: float = 0.01):
        """
        沿放射纤维"迁移"干细胞参数到目标柱
        
        Args:
            stem_params: 干细胞的参数字典（来自 GermCell）
            target_column: 目标柱（Neuroblast 或 SparkColumn）
            noise: 迁移过程中的随机扰动（模拟发育微环境差异）
        """
        with torch.no_grad():
            for name, param in target_column.named_parameters():
                if name in stem_params and param.shape == stem_params[name].shape:
                    guided = stem_params[name] + torch.randn_like(param) * noise
                    param.copy_(guided)
                # 其他参数保持其原有的初始化


class GermCell(nn.Module):
    """
    单个干细胞（放射状胶质细胞）
    
    能力：
    - 保存"母模板"参数（基因）
    - 不对称分裂：spark() → 产生 Neuroblast + 自身保留
    - 对称分裂：replicate() → 产生新 GermCell（扩大干细胞池）
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int,
                 max_len: int = 128, cell_id: int = 0):
        super().__init__()
        self.cell_id = cell_id
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        # 干细胞持有的"基因模板"——一个完整成熟柱的参数
        self.template = SparkColumn(d_model, d_ff, n_heads, max_len, num_layers=4)

        # 放射纤维：用于引导子代初始化
        self.fiber = RadialFiber(d_model, d_ff)

        # 统计
        self.spark_count = 0

    def get_stem_params(self) -> dict:
        """获取干细胞当前参数（作为子代的基因模板）"""
        return {name: param.data.clone()
                for name, param in self.template.named_parameters()}

    def spark(self, noise: float = 0.05) -> Neuroblast:
        """
        不对称分裂：产生一个新的 Neuroblast（只有 0-1 层）
        干细胞自身保留（不改变）
        
        Args:
            noise: 遗传噪声标准差（模拟突变/微环境差异）
        Returns:
            新生的 Neuroblast
        """
        self.spark_count += 1

        # 创建 Neuroblast：初始只有 1 层 (L6)
        neuroblast = Neuroblast(
            self.d_model, self.d_ff, self.n_heads,
            num_layers=1  # 从 L6 开始
        )

        # 沿放射纤维引导初始化
        stem_params = self.get_stem_params()
        self.fiber.guide(stem_params, neuroblast, noise=noise)

        return neuroblast

    def replicate(self) -> 'GermCell':
        """
        对称分裂：复制自身 → 扩大干细胞池
        用于初始化多个 GermCell 时
        """
        new_cell = GermCell(
            self.d_model, self.d_ff, self.n_heads,
            cell_id=self.cell_id + 100  # 给新 ID
        )
        # 复制模板参数
        new_cell.template.load_state_dict(self.template.state_dict())
        return new_cell

    def update_template(self, best_column: SparkColumn):
        """
        用表现最好的皮质柱更新干细胞模板
        模拟"最优基因被保留"的进化压力
        """
        self.template.load_state_dict(best_column.state_dict())


class GerminalPool(nn.Module):
    """
    干细胞池
    
    不参与前向计算。
    负责：
    - 管理多个 GermCell
    - spark() 生成新 Neuroblast
    - 维持干细胞池的健康（对称分裂扩大、用最优柱更新模板）
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int,
                 num_cells: int = 2, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        # 初始化干细胞
        self.cells = nn.ModuleList([
            GermCell(d_model, d_ff, n_heads, max_len, cell_id=i)
            for i in range(num_cells)
        ])

        self.total_sparks = 0

    def spark(self, noise: float = 0.05) -> Neuroblast:
        """
        从干细胞池产生一个新 Neuroblast
        
        策略：从所有干细胞中选 spark 次数最少的（负载均衡）
        """
        # 选负载最低的干细胞
        cell = min(self.cells, key=lambda c: c.spark_count)
        neuroblast = cell.spark(noise=noise)
        self.total_sparks += 1

        print(f"  🔥 GerminalPool.spark() #{self.total_sparks}: "
              f"GermCell[{cell.cell_id}] → Neuroblast (layers=1)")

        return neuroblast

    def expand_pool(self):
        """对称分裂：复制每个干细胞，扩大池子"""
        new_cells = []
        for cell in self.cells:
            new_cells.append(cell.replicate())
        for c in new_cells:
            self.cells.append(c)
        print(f"  🧬 GerminalPool expanded: {len(self.cells)} cells")

    def update_from_best(self, best_column: SparkColumn):
        """用表现最好的柱更新所有干细胞的模板"""
        for cell in self.cells:
            cell.update_template(best_column)
        print(f"  🧬 GerminalPool template updated from best column")


# ═══════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    d_model, d_ff, n_heads = 64, 256, 4

    print("=== GerminalPool 测试 ===")
    pool = GerminalPool(d_model, d_ff, n_heads, num_cells=1)

    # 不对称分裂
    nb = pool.spark(noise=0.05)
    print(f"Neuroblast: layers={nb.num_layers}, params={sum(p.numel() for p in nb.parameters()):,}")

    # 再 spark 一次
    nb2 = pool.spark()
    print(f"Neuroblast2: layers={nb2.num_layers}")

    # 对称分裂扩大池子
    pool.expand_pool()
    print(f"池中有 {len(pool.cells)} 个干细胞")
