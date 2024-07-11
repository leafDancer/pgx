from typing import Optional

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO = jnp.int32(0)
DIM = 5 # mis所对应的knp问题的维度
NPOINT = 1000 # mis的点数
OBS_SPACE = (1,NPOINT,NPOINT) # obs的形状 一个graph bool
SPHERES_SHAPE = (NPOINT,DIM) # 球的点坐标 fp64

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(OBS_SPACE, dtype=jnp.bool_) # (npoint,npoint)
    rewards: Array = jnp.float32([0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(NPOINT, dtype=jnp.bool_) # (npoint,)
    _step_count: Array = jnp.int32(0)
    # --- MIS specific ---
    # npoint x npoint board
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]
    _spheres: Array = jnp.zeros(SPHERES_SHAPE, jnp.float32) # (npoint,dim)
    _solution: Array = jnp.zeros(NPOINT,jnp.bool_)

    @property
    def env_id(self) -> core.EnvId:
        return "MIS"

class PlayMIS(core.Env):
    def __init__(self):
        super().__init__()

    def step(self, state: core.State, action: Array, key: Optional[Array] = None) -> core.State:
        '''assert key is not None, (
            "v2.0.0 changes the signature of step. Please specify PRNGKey at the third argument:\n\n"
            "  * <  v2.0.0: step(state, action)\n"
            "  * >= v2.0.0: step(state, action, key)\n\n"
            "See v2.0.0 release note for more details:\n\n"
            "  https://github.com/sotetsuk/pgx/releases/tag/v2.0.0"
        )'''
        return super().step(state, action)

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array,key) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "MIS"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 1

def _init(rng: PRNGKey) -> State:
    spheres = jax.random.uniform(rng,SPHERES_SHAPE,minval=-1,maxval=1)
    spheres = spheres / jnp.expand_dims(jnp.linalg.norm(spheres,axis=-1),axis=-1)
    return State(_spheres=spheres.ravel())

def _step(state: State, action):
    spheres = state._spheres.reshape((NPOINT,DIM))
    legal_action_mask = state.legal_action_mask.reshape((NPOINT,1))
    spheres = spheres * legal_action_mask
    # 将当前点计入solutions 并更新step count
    #solution = state._solution.at[state._step_count].set(state._spheres[action])
    # 计算出当前点和其他点的角度 找到冲突的点坐标行id 对所有的冲突点坐标置零 得到新的spheres矩阵
    angles = spheres[action] @ spheres.T

    spheres = jnp.where(~jnp.expand_dims((angles>0.5),axis=-1), spheres, 0.)

    # 计算当前的legal action mask
    legal_action_mask = jnp.zeros(NPOINT,jnp.bool_)
    legal_action_mask = jnp.where(~(spheres.sum(axis=-1)>0), legal_action_mask, True)
    # 更新activated
    return state.replace(
        _solution=state._solution.at[action].set(True),
        rewards=jnp.float32([1.0]),
        legal_action_mask=legal_action_mask,
        terminated=~legal_action_mask.any(),
    )

def _observe(state: State, player_id):
    spheres = state._spheres.reshape((NPOINT,DIM))
    legal_action_mask = state.legal_action_mask.reshape((NPOINT,1))
    spheres = spheres * legal_action_mask
    angles = spheres @ spheres.T
    angles = jnp.where(angles<=0.5,angles,0)
    angles = jnp.where(angles>0.5,angles,1)
    return angles.reshape(OBS_SPACE) # (history,NPOINT,NPOINT)

if __name__=='__main__':
    key = jax.random.PRNGKey(3)
    key,subkey = jax.random.split(key)
    keys = jax.random.split(subkey,16)
    state = jax.vmap(_init)(keys)
    while not state.terminated.all():
        key = jax.random.PRNGKey(3)
        #print(jnp.linspace(0,999,num=1000).shape, state.legal_action_mask.shape)
        action = jax.random.choice(key,a=jnp.int32(jnp.linspace(0,999,num=1000)),p=state.legal_action_mask)
        state = _step(state,action)
        print(_observe(state,0).shape)
        #print(sum(state.legal_action_mask))
    print(state._step_count,state._solution.sum())
    assert state._solution.sum() == state._step_count
    sol = state._spheres * state._solution.reshape(-1,1)
    
