import axios from 'axios';
import { io, Socket } from 'socket.io-client';
import dotenv from 'dotenv';

dotenv.config();

const API_URL = process.env.API_URL || 'http://localhost:3000';
const SOCKET_URL = process.env.SOCKET_URL || 'http://localhost:3000';
const ROOM_ID = process.env.ROOM_ID || 'ZZZ000';
const PLAYER_ID = process.env.PLAYER_ID || 'playerB';
const TEAM = process.env.TEAM || 'B';
const AI_HOST = process.env.AI_HOST || 'http://localhost:9000';

interface Action {
  type: 'BASIC_ATTACK' | 'SPECIAL_SKILL';
  sourcePlayerId: string;
  targetPlayerId: string;
  skillId?: string;
}

function extractBattle(payload: any) {
  return payload?.battle ?? payload;
}

async function decideAction(hero: any, enemy: any): Promise<{ action: string; confidence: number }> {
  try {
    const res = await axios.post(`${AI_HOST}/predict`, { hero, enemy, context: {} });
    return res.data;
  } catch {
    // Fallback heurístico
    if (hero.power <= 1) {
      return { action: 'BASICO', confidence: 0 };
    }
    return { action: 'ATAQUE', confidence: 0 };
  }
}

function mapAction(decision: string, targetId: string): Action {
  if (decision === 'SPECIAL') {
    return { type: 'SPECIAL_SKILL', sourcePlayerId: PLAYER_ID, targetPlayerId: targetId, skillId: 'CORTADA' };
  }
  // BASICO o ATAQUE se mapean al ataque básico
  return { type: 'BASIC_ATTACK', sourcePlayerId: PLAYER_ID, targetPlayerId: targetId };
}

export class AIClient2 {
  private socket: Socket;
  private turns: string[] = [];
  private currentTurn: string | null = null;

  constructor(socket?: Socket) {
    this.socket = socket || io(SOCKET_URL);
    this.wire();
  }

  private wire() {
    this.socket.on('connect', async () => {
      await axios.post(`${API_URL}/api/rooms`, { id: ROOM_ID, mode: '1v1', allowAI: false, credits: 100, heroLevel: 1, ownerId: 'ownerB' }).catch(() => {});
      this.socket.emit('joinRoom', { roomId: ROOM_ID, player: { id: PLAYER_ID, heroLevel: 1 } });
      this.socket.emit('playerReady', { roomId: ROOM_ID, playerId: PLAYER_ID, team: TEAM });
    });

    this.socket.on('battleStarted', async (data: any) => {
      this.socket.emit('joinBattle', { roomId: ROOM_ID, playerId: PLAYER_ID });
      const battle = extractBattle(data);
      this.turns = battle?.turns || [];
      this.currentTurn = this.turns[0] || null;
      if (this.currentTurn === PLAYER_ID) {
        await this.playTurn(battle);
      }
    });

    this.socket.on('actionResolved', async (data: any) => {
      const battle = extractBattle(data);
      this.currentTurn = data?.nextTurnPlayer || this.currentTurn;
      if (this.currentTurn === PLAYER_ID) {
        await this.playTurn(battle);
      }
    });
  }

  private async playTurn(battle: any) {
    const players = battle.players || [];
    const me = players.find((p: any) => p.playerId === PLAYER_ID);
    const enemy = players.find((p: any) => p.playerId !== PLAYER_ID);
    if (!me || !enemy) return;
    const decision = await decideAction(me.heroStats.hero, enemy.heroStats.hero);
    const action = mapAction(decision.action, enemy.playerId);
    this.socket.emit('submitAction', { roomId: ROOM_ID, action });
  }
}

if (require.main === module) {
  new AIClient2();
}
