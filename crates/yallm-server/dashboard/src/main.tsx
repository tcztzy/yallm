import '@mantine/core/styles.css';
import './styles.css';

import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Checkbox,
  Container,
  Group,
  MantineProvider,
  NumberFormatter,
  Progress,
  ScrollArea,
  SegmentedControl,
  Select,
  SimpleGrid,
  Stack,
  Table,
  Text,
  TextInput,
  ThemeIcon,
  Title,
  Tooltip as MantineTooltip,
} from '@mantine/core';
import {
  IconActivity,
  IconAlertTriangle,
  IconBolt,
  IconChartBar,
  IconClock,
  IconDatabase,
  IconGauge,
  IconRefresh,
  IconSearch,
  IconServer,
  IconTrash,
} from '@tabler/icons-react';
import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';

type MonitorEvent = {
  request_id: number;
  timestamp_ms: number;
  method: string;
  uri: string;
  endpoint: string;
  status: number;
  latency_ms: number;
  request_bytes: number;
  response_bytes: number;
  provider: string | null;
  model: string | null;
  upstream_model: string | null;
  upstream_url: string | null;
  stream: boolean;
};

type EventsResponse = {
  object: 'list';
  data: MonitorEvent[];
};

type Filters = {
  query: string;
  method: string | null;
  provider: string | null;
  statusGroup: string | null;
  limit: string;
};

type MetricCardProps = {
  label: string;
  value: React.ReactNode;
  detail: string;
  icon: MetricIcon;
  color: string;
  progress?: number;
};

type CountPoint = {
  label: string;
  value: number;
};

type LatencyPoint = {
  time: string;
  p50: number;
  p95: number;
};

const STATUS_COLORS = ['#0f8a5f', '#2563eb', '#b25b00', '#c2413d'];
const metricIcons = {
  activity: IconActivity,
  alert: IconAlertTriangle,
  bolt: IconBolt,
  database: IconDatabase,
  gauge: IconGauge,
  server: IconServer,
} as const;
type MetricIcon = keyof typeof metricIcons;

const limitOptions = ['100', '250', '500', '1000', '2000'].map((value) => ({
  value,
  label: value,
}));
const statusOptions = [
  { value: 'all', label: 'All' },
  { value: '2', label: '2xx' },
  { value: '3', label: '3xx' },
  { value: '4', label: '4xx' },
  { value: '5', label: '5xx' },
];

function App() {
  const [events, setEvents] = useState<MonitorEvent[]>([]);
  const [filters, setFilters] = useState<Filters>({
    query: '',
    method: null,
    provider: null,
    statusGroup: null,
    limit: '500',
  });
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [status, setStatus] = useState('Loading');
  const [loading, setLoading] = useState(false);

  const loadEvents = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/dashboard/api/events?limit=${encodeURIComponent(filters.limit)}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const payload = (await response.json()) as EventsResponse;
      setEvents(Array.isArray(payload.data) ? payload.data : []);
      setStatus(`Updated ${new Date().toLocaleTimeString()}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(`Load failed: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadEvents();
  }, [filters.limit]);

  useEffect(() => {
    if (!autoRefresh) {
      return;
    }
    const timer = window.setInterval(() => {
      void loadEvents();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [autoRefresh, filters.limit]);

  const methodOptions = useMemo(
    () => unique(events.map((event) => event.method).filter(Boolean)).map(toOption),
    [events]
  );
  const providerOptions = useMemo(
    () => unique(events.map(providerName)).map(toOption),
    [events]
  );

  const filtered = useMemo(() => filterEvents(events, filters), [events, filters]);
  const metrics = useMemo(() => buildMetrics(events, filtered), [events, filtered]);
  const latencyData = useMemo(() => bucketByMinute(filtered).slice(-30), [filtered]);
  const statusData = useMemo(() => statusMix(filtered), [filtered]);
  const endpointData = useMemo(() => topCounts(filtered, 'endpoint', 8), [filtered]);
  const providerData = useMemo(() => topCounts(filtered, 'provider', 6, 'local'), [filtered]);
  const health = useMemo(() => healthState(filtered), [filtered]);
  const providerCount = useMemo(() => unique(filtered.map(providerName)).length, [filtered]);
  const upstreamCount = useMemo(
    () => unique(filtered.map((event) => event.upstream_url).filter((value): value is string => Boolean(value))).length,
    [filtered]
  );
  const latestEvent = filtered[0];

  const clearEvents = async () => {
    const response = await fetch('/dashboard/api/events', { method: 'DELETE' });
    if (!response.ok) {
      setStatus(`Clear failed: HTTP ${response.status}`);
      return;
    }
    await loadEvents();
  };

  return (
    <MantineProvider defaultColorScheme="light">
      <div className="dashboard-shell">
        <Container size="2xl" py="lg" className="dashboard-container">
          <Stack gap="lg">
            <section className="dashboard-header">
              <Group justify="space-between" align="flex-start" gap="md" wrap="wrap">
                <div>
                  <Group gap="xs" mb={8}>
                    <Badge color={health.color} variant="light" radius="sm">
                      {health.label}
                    </Badge>
                    <Text c="dimmed" size="sm">
                      {status}
                    </Text>
                  </Group>
                  <Title order={1} className="dashboard-title">
                    yallm Dashboard
                  </Title>
                </div>
                <Group gap="xs" justify="flex-end" className="header-actions">
                  <MantineTooltip label="Refresh events">
                    <ActionIcon
                      aria-label="Refresh events"
                      variant="default"
                      size="lg"
                      loading={loading}
                      onClick={() => void loadEvents()}
                    >
                      <IconRefresh size={18} />
                    </ActionIcon>
                  </MantineTooltip>
                  <Button
                    variant="default"
                    color="red"
                    leftSection={<IconTrash size={16} />}
                    onClick={() => void clearEvents()}
                  >
                    Clear
                  </Button>
                </Group>
              </Group>
              <div className="header-strip">
                <SignalItem icon={<IconDatabase size={16} />} label="Loaded" value={`${formatNumber(events.length)} events`} />
                <SignalItem icon={<IconServer size={16} />} label="Providers" value={formatNumber(providerCount)} />
                <SignalItem icon={<IconChartBar size={16} />} label="Upstreams" value={formatNumber(upstreamCount)} />
                <SignalItem icon={<IconClock size={16} />} label="Latest" value={latestEvent ? fmtTime(latestEvent.timestamp_ms) : 'none'} />
              </div>
            </section>

            <section className="filters-surface">
              <TextInput
                placeholder="Search endpoint, model, upstream"
                value={filters.query}
                onChange={(event) => updateFilter(setFilters, 'query', event.currentTarget.value)}
                leftSection={<IconSearch size={16} />}
                className="search-input"
              />
              <Select
                placeholder="Method"
                data={methodOptions}
                value={filters.method}
                onChange={(value) => updateFilter(setFilters, 'method', value)}
                clearable
                className="compact-select"
              />
              <Select
                placeholder="Provider"
                data={providerOptions}
                value={filters.provider}
                onChange={(value) => updateFilter(setFilters, 'provider', value)}
                clearable
                className="compact-select"
              />
              <SegmentedControl
                data={statusOptions}
                value={filters.statusGroup ?? 'all'}
                onChange={(value) => updateFilter(setFilters, 'statusGroup', value === 'all' ? null : value)}
                className="status-segment"
              />
              <Select
                aria-label="Event limit"
                data={limitOptions}
                value={filters.limit}
                onChange={(value) => updateFilter(setFilters, 'limit', value ?? '500')}
                className="limit-select"
              />
              <Checkbox
                label="Auto refresh"
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.currentTarget.checked)}
                className="auto-refresh"
              />
            </section>

            <SimpleGrid cols={{ base: 1, xs: 2, md: 3, xl: 6 }} spacing="sm">
            {metrics.map((metric) => (
              <MetricCard key={metric.label} {...metric} />
            ))}
          </SimpleGrid>

            <div className="chart-grid">
              <ChartCard title="Latency" detail="p50 / p95">
                <ResponsiveContainer width="100%" height={270}>
                  <LineChart data={latencyData} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                    <CartesianGrid stroke="#e4e9f0" strokeDasharray="4 4" />
                    <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} width={44} unit=" ms" />
                    <RechartsTooltip formatter={(value) => [`${value} ms`, '']} />
                    <Legend />
                    <Line type="monotone" dataKey="p50" stroke="#2563eb" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="p95" stroke="#b25b00" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </ChartCard>
              <ChartCard title="Status Mix" detail={`${formatNumber(filtered.length)} requests`}>
                <ResponsiveContainer width="100%" height={270}>
                  <PieChart>
                    <Pie data={statusData} dataKey="value" nameKey="label" innerRadius={64} outerRadius={94}>
                      {statusData.map((_, index) => (
                        <Cell key={index} fill={STATUS_COLORS[index % STATUS_COLORS.length] ?? '#647184'} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </ChartCard>
              <ChartCard title="Endpoint Volume" detail="top routes">
                <HorizontalBarChart data={endpointData} />
              </ChartCard>
              <ChartCard title="Provider Volume" detail="selected providers">
                <HorizontalBarChart data={providerData} />
              </ChartCard>
            </div>

            <Card withBorder radius="sm" className="panel-card request-panel">
              <Group justify="space-between" mb="sm">
              <Title order={2} size="h4">
                Recent Requests
              </Title>
              <Badge variant="light" color="gray" radius="sm">
                <NumberFormatter value={filtered.length} thousandSeparator /> shown
              </Badge>
            </Group>
            <ScrollArea>
              <Table highlightOnHover verticalSpacing="sm" miw={1240} className="requests-table">
                <Table.Thead className="sticky-head">
                  <Table.Tr>
                    <Table.Th>Time</Table.Th>
                    <Table.Th>Status</Table.Th>
                    <Table.Th>Method</Table.Th>
                    <Table.Th>Endpoint</Table.Th>
                    <Table.Th>Provider</Table.Th>
                    <Table.Th>Model</Table.Th>
                    <Table.Th>Upstream</Table.Th>
                    <Table.Th>Latency</Table.Th>
                    <Table.Th>Bytes</Table.Th>
                    <Table.Th>Stream</Table.Th>
                    <Table.Th>Request</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {filtered.length === 0 ? (
                    <Table.Tr>
                      <Table.Td colSpan={11}>
                        <Text c="dimmed" ta="center" py="xl">
                          No request events
                        </Text>
                      </Table.Td>
                    </Table.Tr>
                  ) : (
                    filtered.slice(0, 250).map((event) => <EventRow key={event.request_id} event={event} />)
                  )}
                </Table.Tbody>
              </Table>
            </ScrollArea>
          </Card>
          </Stack>
        </Container>
      </div>
    </MantineProvider>
  );
}

function SignalItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="signal-item">
      <span className="signal-icon">{icon}</span>
      <span className="signal-label">{label}</span>
      <span className="signal-value">{value}</span>
    </div>
  );
}

function MetricCard({ label, value, detail, icon, color, progress }: MetricCardProps) {
  const Icon = metricIcons[icon];
  return (
    <Card withBorder radius="sm" className="metric-card">
      <Group justify="space-between" align="flex-start" mb={8}>
        <Text c="dimmed" size="xs" fw={700} tt="uppercase">
          {label}
        </Text>
        <ThemeIcon color={color} variant="light" radius="sm" size="sm">
          <Icon size={14} />
        </ThemeIcon>
      </Group>
      <Text className="metric-value">{value}</Text>
      <Text c="dimmed" size="xs" mt={6}>
        {detail}
      </Text>
      {progress !== undefined ? <Progress value={progress} color={color} size="xs" radius="xs" mt="sm" /> : null}
    </Card>
  );
}

function ChartCard({ title, detail, children }: { title: string; detail: string; children: React.ReactNode }) {
  return (
    <Card withBorder radius="sm" className="panel-card">
      <Group justify="space-between" align="center" mb="sm">
        <Title order={2} size="h4">
          {title}
        </Title>
        <Text c="dimmed" size="xs" fw={700} tt="uppercase">
          {detail}
        </Text>
      </Group>
      {children}
    </Card>
  );
}

function HorizontalBarChart({ data }: { data: CountPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} layout="vertical" margin={{ left: 16, right: 24 }}>
        <CartesianGrid stroke="#e4e9f0" strokeDasharray="4 4" />
        <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
        <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12 }} />
        <RechartsTooltip />
        <Bar dataKey="value" fill="#2563eb" radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function EventRow({ event }: { event: MonitorEvent }) {
  return (
    <Table.Tr>
      <Table.Td className="mono">{fmtTime(event.timestamp_ms)}</Table.Td>
      <Table.Td>
        <Badge color={statusColor(event.status)} variant="light" radius="sm">
          {event.status}
        </Badge>
      </Table.Td>
      <Table.Td>
        <Badge color="gray" variant="outline" radius="sm" className="method-badge">
          {event.method}
        </Badge>
      </Table.Td>
      <Table.Td className="mono">{event.endpoint}</Table.Td>
      <Table.Td>
        <Badge color={event.provider ? 'blue' : 'gray'} variant="light" radius="sm">
          {providerName(event)}
        </Badge>
      </Table.Td>
      <Table.Td className="mono">{event.model || ''}</Table.Td>
      <Table.Td className="mono url-cell" title={event.upstream_url || ''}>
        {event.upstream_url || <span className="muted-cell">local</span>}
      </Table.Td>
      <Table.Td>
        <NumberFormatter value={event.latency_ms} thousandSeparator suffix=" ms" />
      </Table.Td>
      <Table.Td>{fmtBytes((event.request_bytes || 0) + (event.response_bytes || 0))}</Table.Td>
      <Table.Td>
        <Badge color={event.stream ? 'teal' : 'gray'} variant="light" radius="sm">
          {event.stream ? 'yes' : 'no'}
        </Badge>
      </Table.Td>
      <Table.Td className="mono">{event.request_id}</Table.Td>
    </Table.Tr>
  );
}

function updateFilter<K extends keyof Filters>(
  setFilters: React.Dispatch<React.SetStateAction<Filters>>,
  key: K,
  value: Filters[K]
) {
  setFilters((current) => ({ ...current, [key]: value }));
}

function toOption(value: string) {
  return { value, label: value };
}

function unique(values: string[]) {
  return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
}

function providerName(event: MonitorEvent) {
  return event.provider || 'local';
}

function filterEvents(events: MonitorEvent[], filters: Filters) {
  const query = filters.query.trim().toLowerCase();
  return events.filter((event) => {
    if (filters.method && event.method !== filters.method) {
      return false;
    }
    if (filters.provider && providerName(event) !== filters.provider) {
      return false;
    }
    if (filters.statusGroup && String(Math.floor(event.status / 100)) !== filters.statusGroup) {
      return false;
    }
    if (!query) {
      return true;
    }
    return [
      event.endpoint,
      event.uri,
      event.model,
      event.upstream_model,
      event.upstream_url,
      event.provider,
      String(event.request_id),
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
      .includes(query);
  });
}

function buildMetrics(allEvents: MonitorEvent[], events: MonitorEvent[]): MetricCardProps[] {
  const total = events.length;
  const ok = events.filter((event) => event.status >= 200 && event.status < 300).length;
  const clientErrors = events.filter((event) => event.status >= 400 && event.status < 500).length;
  const serverErrors = events.filter((event) => event.status >= 500).length;
  const errors = clientErrors + serverErrors;
  const avgLatency = total ? Math.round(events.reduce((sum, event) => sum + event.latency_ms, 0) / total) : 0;
  const p95 = percentile(
    events.map((event) => event.latency_ms),
    0.95
  );
  const bytes = events.reduce((sum, event) => sum + event.request_bytes + event.response_bytes, 0);
  const streaming = events.filter((event) => event.stream).length;
  const successRate = total ? Math.round((ok / total) * 100) : 0;
  const errorRate = total ? Math.round((errors / total) * 100) : 0;
  const streamingRate = total ? Math.round((streaming / total) * 100) : 0;

  return [
    {
      label: 'Requests',
      value: formatNumber(total),
      detail: `${formatNumber(allEvents.length)} loaded`,
      icon: 'activity',
      color: 'blue',
    },
    {
      label: 'Success',
      value: `${successRate}%`,
      detail: `${formatNumber(ok)} 2xx`,
      icon: 'gauge',
      color: successRate >= 95 || total === 0 ? 'green' : 'orange',
      progress: successRate,
    },
    {
      label: 'Errors',
      value: formatNumber(errors),
      detail: `${formatNumber(clientErrors)} 4xx / ${formatNumber(serverErrors)} 5xx`,
      icon: 'alert',
      color: errors ? 'red' : 'green',
      progress: errorRate,
    },
    {
      label: 'p95 Latency',
      value: `${formatNumber(p95)} ms`,
      detail: `avg ${formatNumber(avgLatency)} ms`,
      icon: 'bolt',
      color: 'orange',
    },
    {
      label: 'Traffic',
      value: fmtBytes(bytes),
      detail: 'request + response',
      icon: 'database',
      color: 'teal',
    },
    {
      label: 'Streaming',
      value: `${streamingRate}%`,
      detail: `${formatNumber(streaming)} streamed`,
      icon: 'server',
      color: 'cyan',
      progress: streamingRate,
    },
  ];
}

function healthState(events: MonitorEvent[]) {
  if (!events.length) {
    return { label: 'Idle', color: 'gray' };
  }
  if (events.some((event) => event.status >= 500)) {
    return { label: '5xx Active', color: 'red' };
  }
  if (events.some((event) => event.status >= 400)) {
    return { label: '4xx Active', color: 'orange' };
  }
  return { label: 'Healthy', color: 'green' };
}

function statusMix(events: MonitorEvent[]): CountPoint[] {
  return [
    { label: '2xx', value: events.filter((event) => event.status >= 200 && event.status < 300).length },
    { label: '3xx', value: events.filter((event) => event.status >= 300 && event.status < 400).length },
    { label: '4xx', value: events.filter((event) => event.status >= 400 && event.status < 500).length },
    { label: '5xx', value: events.filter((event) => event.status >= 500).length },
  ].filter((item) => item.value > 0);
}

function topCounts(
  events: MonitorEvent[],
  field: 'endpoint' | 'provider',
  limit: number,
  fallback = ''
): CountPoint[] {
  const counts = new Map<string, number>();
  events.forEach((event) => {
    const key = field === 'provider' ? event.provider || fallback : event.endpoint;
    if (key) {
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  });
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([label, value]) => ({ label, value }));
}

function bucketByMinute(events: MonitorEvent[]): LatencyPoint[] {
  const buckets = new Map<number, number[]>();
  events
    .slice()
    .reverse()
    .forEach((event) => {
      const key = Math.floor(event.timestamp_ms / 60000) * 60000;
      const bucket = buckets.get(key) || [];
      bucket.push(event.latency_ms);
      buckets.set(key, bucket);
    });
  return Array.from(buckets.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([time, values]) => ({
      time: new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      p50: percentile(values, 0.5),
      p95: percentile(values, 0.95),
    }));
}

function percentile(values: number[], p: number) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) {
    return 0;
  }
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * p) - 1);
  return sorted[index] ?? 0;
}

function statusColor(status: number) {
  if (status >= 500) {
    return 'red';
  }
  if (status >= 400) {
    return 'orange';
  }
  if (status >= 200 && status < 300) {
    return 'green';
  }
  return 'gray';
}

function fmtTime(ms: number) {
  if (!ms) {
    return '';
  }
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtBytes(value: number) {
  const bytes = Number(value || 0);
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat().format(value || 0);
}

createRoot(document.getElementById('root') as HTMLElement).render(<App />);
