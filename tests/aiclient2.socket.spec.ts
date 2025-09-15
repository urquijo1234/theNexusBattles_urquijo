import { EventEmitter } from 'events';
import assert from 'assert';
import axios from 'axios';
import { AIClient2 } from '../src/clients/AIClient2';

class MockSocket extends EventEmitter {
  emit(event: string, payload: any) {
    return super.emit(event, payload);
  }
}

(async () => {
  (axios.post as any) = async () => { throw new Error('fail'); };
  const socket = new MockSocket();
  const actions: any[] = [];
  socket.emit = (event: string, payload: any) => {
    if (event === 'submitAction') actions.push(payload);
    return EventEmitter.prototype.emit.call(socket, event, payload);
  };
  new AIClient2(socket as any);
  const battle = { turns: ['playerB', 'playerA'], players: [
    { playerId: 'playerB', heroStats: { hero: { power: 0 } } },
    { playerId: 'playerA', heroStats: { hero: { power: 5 } } },
  ] };
  socket.emit('battleStarted', battle);
  await new Promise(r => setTimeout(r, 50));
  assert.strictEqual(actions[0].action.type, 'BASIC_ATTACK');
  console.log('AIClient2 test passed');
})();
