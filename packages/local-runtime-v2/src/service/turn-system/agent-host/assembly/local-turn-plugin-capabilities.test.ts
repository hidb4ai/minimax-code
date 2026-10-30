import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { setImmediate as nextEventLoopTurn } from 'node:timers/promises';
import { buildLocalTurnToolCatalog } from './local-turn-tool-catalog.js';
import { Type } from '@sinclair/typebox';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { RuntimeTool } from '@mavis/agent-core/tools';
import type { DatabaseLike } from '@mavis/local-runtime';
import type { McpToolInfo } from '@mavis/mcp';
import { migratePluginTestDatabase } from '../../../../../test/helpers/plugin-database.js';
import { PluginSystem } from '../../../plugin-system/plugin-system.js';
import { PluginMcpRuntime, type PluginMcpPool } from '../../../plugin-system/mcp/runtime.js';
import { SqlitePluginRepository } from '../../../plugin-system/plugin/runtime/repository.js';
import { AgentHostTurnCapabilityLifecycle } from './turn-capability-lifecycle.js';
import {
  detectPluginReferencesForMessages,
  buildPluginReferenceReminder,
} from '@mavis/system-reminder';
import {
  mergeAgentHostTurnCapabilities,
  renderPluginAgentRoster,
  type MergeAgentHostTurnCapabilitiesInput,
} from './local-turn-plugin-capabilities.js';

function tool(name: string): RuntimeTool {
  return {
    def: { name, description: 'Synthetic test tool', schema: Type.Object({}) },
    impl: {
      execute: vi.fn(async () => ({
        tool_name: name,
        text: 'ok',
        content: [{ type: 'text' as const, text: 'ok' }],
      })),
    },
  };
}
function input(
  userText = '[@My Notes](plugin://notes%40local) summarize',
): MergeAgentHostTurnCapabilitiesInput {
  const mcp = tool('mcp__notes__read');
  const app = tool('app_notes_search');
  return {
    tools: [],
    plan: { deferred: false, inlineTools: [] },
    model: { provider: 'synthetic', id: 'offline', contextWindow: 128000 },
    options: {
      enabled: true,
      modelWhitelist: [],
      thresholdPct: 10,
      minDeferCount: 1,
      topKDefault: 5,
      topKMax: 10,
      systemHint: true,
      maxSchemaTextLen: 1000,
    },
    effectivePluginSkills: [{ pluginName: 'notes', name: 'notes:organize' }],
    capabilities: {
      revision: 'r1',
      plugins: [{ name: 'notes', source: 'local', appProviders: ['notes-app'] }],
      skills: [],
      runtimeTools: [mcp, app],
      runtimeToolBindings: [
        { kind: 'mcp', source: 'notes-server', pluginName: 'notes', tool: mcp },
        { kind: 'app', source: 'notes-app', toolMode: 'tool_search', tool: app },
      ],
    },
    userText,
  };
}

describe('Plugin identity selection at the final Runtime capability boundary', () => {
  it('selects exact IDs independently of aliases and never substitutes a same-named source', () => {
    const local = {
      name: 'notes',
      pluginId: 'notes@local',
      skills: ['notes:organize'],
      appTools: [],
      mcpTools: [],
    };
    const official = { ...local, pluginId: 'notes@official' };
    expect(
      detectPluginReferencesForMessages([], '[@Alias](plugin://notes%40local)', [official, local]),
    ).toEqual([local]);
    expect(
      detectPluginReferencesForMessages([], '[@notes](plugin://notes%40missing)', [local]),
    ).toEqual([]);
    expect(detectPluginReferencesForMessages([], '@notes summarize', [official, local])).toEqual(
      [],
    );
    expect(detectPluginReferencesForMessages([], '@notes summarize', [local])).toEqual([local]);
  });

  it('advertises only effective skills, direct MCP and searchable App tools, preserving callable implementations', async () => {
    const request = input();
    const result = mergeAgentHostTurnCapabilities(request);
    expect(result.reminder).toContain('notes:organize');
    expect(result.reminder).toContain('mcp__notes__read');
    expect(result.reminder).toContain('app_notes_search');
    expect(result.reminder).toContain('tool_search');
    expect(result.plan.deferred).toBe(true);
    const mcp = result.tools.find((tool) => tool.def.name === 'mcp__notes__read')!;
    await expect(mcp.impl.execute({ sessionId: 's', turnId: 't' }, {})).resolves.toMatchObject({
      text: 'ok',
    });
    expect(mcp.toolCallProvenanceResolver?.({ phase: 'start', toolName: mcp.def.name })).toEqual([
      expect.objectContaining({ plugin_name: 'notes' }),
    ]);
  });

  it('discovers and invokes deferred MCP and App tools through the final catalog', async () => {
    const request = input();
    const mcp = request.capabilities!.runtimeTools[0]!;
    const result = buildLocalTurnToolCatalog({
      sessionId: 's',
      llmModel: request.model,
      sources: {
        nativeTools: [],
        mcpEntries: [{ tool: mcp, source: 'configured', serverName: 'notes-server' }],
        threadGoalTools: [],
        cuRuntimeAvailable: false,
      },
      config: { ...request.options, modelWhitelist: ['*'], thresholdPct: 0 },
      desktopCapabilities: request.capabilities,
      effectivePluginSkills: request.effectivePluginSkills,
      userText: request.userText,
    });
    expect(result.tools.map((tool) => tool.def.name)).not.toContain(mcp.def.name);
    expect(result.userPromptPrefix).toContain(
      'server `notes-server` via `tool_search` + `mcp_invoke`',
    );
    const search = result.tools.find((tool) => tool.def.name === 'tool_search')!;
    const invoke = result.tools.find((tool) => tool.def.name === 'mcp_invoke')!;
    for (const name of ['mcp__notes__read', 'app_notes_search']) {
      const found = await search.impl.execute({ sessionId: 's', turnId: 't' }, { query: name });
      expect(JSON.stringify(found)).toContain(name);
      await expect(
        invoke.impl.execute({ sessionId: 's', turnId: 't' }, { tool_name: name, arguments: {} }),
      ).resolves.toMatchObject({ text: 'ok' });
    }
  });

  it('reports a plugin disabled before this turn and does not activate it or use another source', () => {
    const request = input();
    const result = mergeAgentHostTurnCapabilities({
      ...request,
      effectivePluginSkills: [],
      capabilities: {
        revision: 'r2',
        plugins: [],
        runtimeTools: [],
        runtimeToolBindings: [],
        skills: [],
      },
    });
    expect(result.reminder).toContain('notes@local');
    expect(result.reminder).toContain('unavailable for this turn');
    expect(result.tools).toEqual([]);
    expect(
      mergeAgentHostTurnCapabilities({ ...request, capabilities: undefined }).reminder,
    ).toContain('unavailable');
    expect(
      mergeAgentHostTurnCapabilities(input('[@notes](plugin://notes%40official)')).reminder,
    ).toContain('notes@official` is unavailable');
  });

  it('bounds capability instructions even for large inventories', () => {
    const result = buildPluginReferenceReminder(
      Array.from({ length: 40 }, (_, index) => ({
        name: `plugin-${index}`,
        pluginId: `plugin-${index}@local`,
        skills: Array.from({ length: 200 }, (_, i) => `plugin-${index}:skill-${i}`),
        appTools: [],
        mcpTools: [],
      })),
    );
    expect(result!.length).toBeLessThan(16384);
    expect(result).toContain('omitted');
    expect(result).toContain('</system-reminder>');
  });
});

function baseInput(userText: string): MergeAgentHostTurnCapabilitiesInput {
  return {
    tools: [],
    plan: { deferred: false, inlineTools: [] },
    model: { provider: "minimax", id: "model", contextWindow: 100_000 },
    options: {
      enabled: true,
      modelWhitelist: [],
      thresholdPct: 0.15,
      minDeferCount: 1,
      topKDefault: 5,
      topKMax: 20,
      systemHint: true,
      maxSchemaTextLen: 4_096,
    },
    effectivePluginSkills: [{ pluginName: "demo", name: "demo:skill" }],
    userText,
  };
}

function view(overrides: Record<string, unknown>) {
  return {
    revision: "revision-1",
    plugins: [
      { name: "demo", version: "1.0.0", source: "local", appProviders: ["notion"] },
    ],
    skills: [],
    runtimeTools: [],
    runtimeToolBindings: [],
    ...overrides,
  } as never;
}

function withDiscovery(names: readonly string[]): string[] {
  return [
    ...names.filter((name) => name !== 'tool_search' && name !== 'mcp_invoke'),
    'tool_search',
    'mcp_invoke',
  ];
}

describe('Plugin Agent roster reminder', () => {
  const agent = (
    scopedName: string,
    description: string,
    origin: 'official' | 'local' = 'local',
  ) => ({
    scopedName,
    pluginName: scopedName.split(':')[0]!,
    agentName: scopedName.split(':').slice(1).join(':'),
    sourceFormat: 'minimax' as const,
    origin,
    packageDigest: 'sha256:demo',
    pluginRoot: '/plugins/demo',
    description,
    systemPrompt: 'Body',
  });

  it('lists sorted scoped names with flattened, bounded descriptions when task is present', () => {
    const result = mergeAgentHostTurnCapabilities({
      ...baseInput('inspect'),
      tools: [tool('read'), tool('task')],
      capabilities: view({
        agents: [
          agent('zeta:writer', 'Writes <docs>\nand notes'),
          agent('alpha:review', 'x'.repeat(300)),
        ],
      }),
    });

    expect(result.reminder).toContain('<plugin-agents>');
    const roster = result.reminder!.slice(result.reminder!.indexOf('<plugin-agents>'));
    expect(roster.indexOf('`alpha:review`')).toBeLessThan(roster.indexOf('`zeta:writer`'));
    expect(roster).toContain('- `zeta:writer`: Writes &lt;docs&gt; and notes');
    expect(roster).toContain(`- \`alpha:review\`: ${'x'.repeat(199)}…`);
  });

  it('omits the roster when this Turn cannot call task', () => {
    expect(
      renderPluginAgentRoster([agent('demo:review', 'Review')], [tool('read')]),
    ).toBeUndefined();
    expect(renderPluginAgentRoster([], [tool('task')])).toBeUndefined();
  });

  it('treats a mentioned Subagent-only Plugin as available only when task can delegate', () => {
    const input = {
      ...baseInput('[@Agents](plugin://agents-only%40local) review this'),
      capabilities: view({ agents: [agent('agents-only:reviewer', 'Reviews')] }),
    };
    const delegable = mergeAgentHostTurnCapabilities({
      ...input,
      tools: [tool('read'), tool('task')],
    });
    expect(delegable.reminder).not.toContain('is unavailable for this turn');
    expect(delegable.reminder).toContain('`agents-only:reviewer`');

    const noTask = mergeAgentHostTurnCapabilities({ ...input, tools: [tool('read')] });
    expect(noTask.reminder).toContain('agents-only@local` is unavailable for this turn');
  });

  it('matches a mentioned Subagent-only official Plugin by its official id', () => {
    const mention = baseInput('[@Review](plugin://review%40official) review this');
    const official = mergeAgentHostTurnCapabilities({
      ...mention,
      tools: [tool('read'), tool('task')],
      capabilities: view({ agents: [agent('review:checker', 'Checks', 'official')] }),
    });
    expect(official.reminder).not.toContain('is unavailable for this turn');
    expect(official.reminder).toContain('`review:checker`');

    const localOnly = mergeAgentHostTurnCapabilities({
      ...mention,
      tools: [tool('read'), tool('task')],
      capabilities: view({ agents: [agent('review:checker', 'Checks')] }),
    });
    expect(localOnly.reminder).toContain('review@official` is unavailable for this turn');
  });
});

function controlledPromise<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((accept, fail) => {
    resolve = accept;
    reject = fail;
  });
  return { promise, resolve, reject };
}

const pluginInventory: McpToolInfo[] = [
  { name: 'list_scopes', description: 'List synthetic scopes', inputSchema: { type: 'object' } },
];
const readinessCleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
  vi.useRealTimers();
  for (const cleanup of readinessCleanups.splice(0).reverse()) await cleanup();
});

async function pluginReadinessFixture({ mcp = true, secondHealthyServer = false } = {}) {
  const dataDir = await mkdtemp(path.join(tmpdir(), 'plugin-turn-readiness-'));
  const pluginRoot = path.join(dataDir, 'plugins', 'notes');
  const manifestPath = path.join(pluginRoot, '.minimax-plugin', 'plugin.json');
  const mcpPath = path.join(pluginRoot, 'notes.mcp.json');
  const manifest = {
    schemaVersion: 1,
    name: 'notes',
    version: '1.0.0',
    description: 'Synthetic scope tools',
    author: 'Test fixture',
    icon: 'icon.png',
    category: 'Other',
    exampleQueries: [],
    apps: [],
    skills: ['skills/notes/SKILL.md'],
    mcpServers: mcp ? ['notes.mcp.json'] : [],
  };
  await mkdir(path.dirname(manifestPath), { recursive: true });
  await mkdir(path.join(pluginRoot, 'skills', 'notes'), { recursive: true });
  await writeFile(manifestPath, JSON.stringify(manifest));
  await writeFile(path.join(pluginRoot, 'icon.png'), Buffer.from([0]));
  await writeFile(path.join(pluginRoot, 'skills', 'notes', 'SKILL.md'),
    '---\nname: notes\ndescription: Find synthetic scopes\n---\nUse the scope tools.\n');
  const writeEndpoint = (url: string) => writeFile(mcpPath, JSON.stringify({
    schemaVersion: 1,
    mcpServers: {
      notes: { type: 'streamable-http', url },
      ...(secondHealthyServer ? { healthy: { type: 'streamable-http', url: 'http://127.0.0.1:8125/mcp' } } : {}),
    },
  }));
  if (mcp) await writeEndpoint('http://127.0.0.1:8123/mcp');

  const Database = createRequire(import.meta.url)('better-sqlite3') as new (filename: string) => DatabaseLike;
  const database = new Database(':memory:');
  const repository = new SqlitePluginRepository(migratePluginTestDatabase(database));
  let inventory = controlledPromise<McpToolInfo[]>();
  let discoveryStarted = controlledPromise<void>();
  const inventories = [inventory];
  let onChange!: () => Promise<void>;
  const pool: PluginMcpPool = {
    listTools: async (serverName) => {
      if (serverName.includes('healthy')) return [{ ...pluginInventory[0]!, name: 'healthy_scopes' }];
      discoveryStarted.resolve();
      return inventory.promise;
    },
    callTool: async () => ({ content: [{ type: 'text', text: 'synthetic scope result' }] }),
    disconnect: async () => undefined,
    shutdown: async () => undefined,
  };
  const createRuntime = () => new PluginMcpRuntime({ pool });
  const system = new PluginSystem({
    dataDir,
    officialCacheRoot: path.join(dataDir, 'official'),
    repository,
    authContextGetter: () => undefined,
    deploymentGetter: () => 'synthetic',
    listReservations: async () => ({ skillNames: [], mcpServerNames: [], toolNames: [] }),
    mcpRuntime: createRuntime(),
    mcpRuntimeFactory: createRuntime,
    localDirectoryWatcherFactory: (changed) => {
      onChange = changed;
      return { start: async () => undefined, whenIdle: async () => undefined, close: () => undefined };
    },
  });
  const lifecycle = new AgentHostTurnCapabilityLifecycle();
  const publication = lifecycle.attach(system);
  let publicationBarrier: ReturnType<typeof controlledPromise<void>> | undefined;
  let publicationStarted = controlledPromise<void>();
  system.attachPublicationPort({
    get activeTurnCount() { return publication.activeTurnCount; },
    tryPublish: (publish) => publication.tryPublish(async () => {
      if (publicationBarrier) {
        publicationStarted.resolve();
        await publicationBarrier.promise;
      }
      await publish();
    }),
  });
  const leases: Array<{ release(): boolean }> = [];
  readinessCleanups.push(async () => {
    inventories.forEach((pending) => pending.resolve(pluginInventory));
    publicationBarrier?.resolve();
    leases.forEach((lease) => lease.release());
    await system.dispose();
    database.close();
    await rm(dataDir, { recursive: true, force: true });
  });
  await system.initialize();
  await system.waitUntilReady();
  expect(system.currentSnapshot.turnCapabilities.plugins).toContainEqual(expect.objectContaining({ name: 'notes' }));
  if (mcp) await discoveryStarted.promise;
  return {
    system,
    async acquire(sessionId = 'session', signal?: AbortSignal) {
      const lease = await lifecycle.acquire({ sessionId, turnId: `turn-${leases.length}`, signal });
      leases.push(lease);
      return lease;
    },
    finishDiscovery() { inventory.resolve(pluginInventory); },
    failDiscovery() { inventory.reject(new Error('Synthetic inventory failure')); },
    blockPublication() {
      publicationBarrier = controlledPromise<void>();
      publicationStarted = controlledPromise<void>();
      return { started: publicationStarted.promise, release: () => publicationBarrier!.resolve() };
    },
    async changeEndpoint() {
      inventory = controlledPromise<McpToolInfo[]>();
      inventories.push(inventory);
      discoveryStarted = controlledPromise<void>();
      await writeEndpoint('http://127.0.0.1:8124/mcp');
      await onChange();
      await discoveryStarted.promise;
    },
    async changeDescription() {
      await writeFile(manifestPath, JSON.stringify({ ...manifest, description: 'Updated notes' }));
      await onChange();
    },
  };
}

describe('Plugin MCP readiness before the first turn capability capture', () => {
  it('waits for cold discovery and captures callable tools on the first turn', async () => {
    const fixture = await pluginReadinessFixture();
    let admitted = false;
    const pending = fixture.acquire().then((lease) => { admitted = true; return lease; });
    await nextEventLoopTurn();
    expect(admitted).toBe(false);
    fixture.finishDiscovery();
    const lease = await pending;
    const scopeTool = lease.capabilities!.runtimeTools.find((entry) => entry.def.name.endsWith('list_scopes'));
    expect(scopeTool).toBeDefined();
    await expect(scopeTool!.impl.execute({ sessionId: 'session', turnId: 'turn' }, {}))
      .resolves.toMatchObject({ text: 'synthetic scope result' });
  });

  it('does not capture a cached empty inventory between discovery and publication', async () => {
    const fixture = await pluginReadinessFixture();
    const publication = fixture.blockPublication();
    let prepared = false;
    const pending = fixture.system.prepareTurnCapabilities().then((value) => { prepared = true; return value; });
    fixture.finishDiscovery();
    await publication.started;
    await nextEventLoopTurn();
    expect(prepared).toBe(false);
    publication.release();
    const preparation = await pending;
    expect(fixture.system.captureTurnCapabilities(preparation).runtimeTools.map((entry) => entry.def.name))
      .toEqual([expect.stringMatching(/list_scopes$/u)]);
  });

  it('cancels one waiter without cancelling discovery needed by another session', async () => {
    const fixture = await pluginReadinessFixture();
    const controller = new AbortController();
    const cancelled = fixture.acquire('cancelled-session', controller.signal);
    const waiting = fixture.acquire('other-session');
    const reason = new Error('Turn cancelled');
    const rejected = expect(cancelled).rejects.toBe(reason);
    controller.abort(reason);
    await rejected;
    fixture.finishDiscovery();
    const lease = await waiting;
    expect(lease.capabilities!.runtimeTools.map((entry) => entry.def.name))
      .toEqual([expect.stringMatching(/list_scopes$/u)]);
  });

  it('keeps a healthy server available after another inventory fails without waiting for retries', async () => {
    const fixture = await pluginReadinessFixture({ secondHealthyServer: true });
    const pending = fixture.acquire();
    fixture.failDiscovery();
    const lease = await pending;
    expect(lease.capabilities!.runtimeTools.map((entry) => entry.def.name))
      .toEqual([expect.stringMatching(/healthy_scopes$/u)]);
    expect(fixture.system.currentSnapshot.diagnostics)
      .toContainEqual(expect.objectContaining({ code: 'MCP_SERVER_LIST_FAILED' }));
  });

  it('bounds an unpublished inventory wait by the turn preparation deadline', async () => {
    const fixture = await pluginReadinessFixture();
    const publication = fixture.blockPublication();
    fixture.finishDiscovery();
    await publication.started;
    vi.useFakeTimers();
    const pending = fixture.system.prepareTurnCapabilities();
    const rejected = expect(pending).rejects.toMatchObject({ code: 'MCP_INVENTORY_NOT_READY' });
    await vi.advanceTimersByTimeAsync(20_001);
    await rejected;
    publication.release();
  });

  it('rejects a newly warming endpoint at final capture after preparation', async () => {
    const fixture = await pluginReadinessFixture();
    fixture.finishDiscovery();
    const preparation = await fixture.system.prepareTurnCapabilities();
    expect(fixture.system.captureTurnCapabilities(preparation).runtimeTools).toHaveLength(1);
    await fixture.changeEndpoint();
    expect(() => fixture.system.captureTurnCapabilities(preparation))
      .toThrow(expect.objectContaining({ code: 'MCP_INVENTORY_NOT_READY' }));
    fixture.finishDiscovery();
    const lease = await fixture.acquire();
    expect(lease.capabilities!.runtimeTools).toHaveLength(1);
  });

  it('reuses valid inventory while a previous turn retains its own generation', async () => {
    const fixture = await pluginReadinessFixture();
    fixture.finishDiscovery();
    const first = await fixture.acquire();
    await fixture.changeDescription();
    const second = await fixture.acquire('second-session');
    expect(second.capabilities!.runtimeTools.map((entry) => entry.def.name))
      .toEqual(first.capabilities!.runtimeTools.map((entry) => entry.def.name));
    expect(second.capabilities!.revision).not.toBe(first.capabilities!.revision);
    await expect(first.capabilities!.runtimeTools[0]!.impl.execute({ sessionId: 'session', turnId: 'turn' }, {}))
      .resolves.toMatchObject({ text: 'synthetic scope result' });
  });

  it('admits a skill-only plugin without requiring MCP discovery', async () => {
    const fixture = await pluginReadinessFixture({ mcp: false });
    const lease = await fixture.acquire();
    expect(lease.capabilities!.runtimeTools).toEqual([]);
    expect(lease.capabilities!.skills).toEqual([expect.objectContaining({ name: expect.stringContaining('notes') })]);
  });
});
