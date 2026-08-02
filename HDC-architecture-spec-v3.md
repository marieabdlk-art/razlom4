# HDC Cognitive Architecture v3

## Полная референсная спецификация реконструируемой системы

**Статус:** implementable reference interpretation  
**Язык спецификации:** русский  
**Целевая реализация:** Python 3.12, asyncio, NumPy, опционально PyTorch  
**Версия формата состояния:** `hdc-ca/3.0`

> Этот документ не выдаёт додуманные детали за слова автора исходной системы.
> Метки ниже разделяют происхождение требований:
>
> - **[ИСТОЧНИК]** — прямо следует из собранных описаний автора;
> - **[ИНТЕРПРЕТАЦИЯ]** — наиболее естественная техническая трактовка;
> - **[РЕФЕРЕНС]** — решение, добавленное для полноты, воспроизводимости и
>   безопасности реализации.

---

## 1. Назначение и границы утверждений

Система является ограниченным онлайновым когнитивным симулятором на основе
Hyperdimensional Computing / Vector Symbolic Architecture. Она принимает
текстовые и числовые сенсорные события, поддерживает изменяемое внутреннее
состояние, ассоциативную память, непрерывные регуляторы, цели и выбор действия.

Первая реализация должна подтверждать только следующие свойства:

1. онлайновое приобретение ассоциаций без предварительного обучения весов;
2. различение новизны, повторного наблюдения и практического применения;
3. ограниченная ассоциативная генерация или предсказание следующего символа;
4. влияние ресурсов и регуляторов на маршрутизацию вычислений и памяти;
5. воспроизводимое состояние при фиксированных конфигурации и журнале;
6. безопасная, версионируемая консолидация STM → LTM;
7. измеримое прекращение циклов без общего выключения системы.

Система **не заявляет** понимание естественного языка, сознание, биологическую
правдоподобность, AGI или превосходство над трансформерами. Биологические имена
гормонов являются названиями инженерных регуляторов, пока отдельный эксперимент
не докажет более сильную аналогию.

---

## 2. Профиль воспроизводимости

### 2.1 Обязательная конфигурация

```yaml
spec_version: hdc-ca/3.0
global_seed: <256-bit hex>
vector_dimension: 8192
vector_accumulator_bits: 16
fixed_point_fraction_bits: 16
unicode_normalization: NFC
tokenizer_version: char-ngram-3-5/v1
max_signal_hops: 64
max_events_per_tick: 256
stm_capacity: 4096
ltm_capacity: 65536
event_log_capacity: 1000000
snapshot_retention: 3
```

Размерность может принимать значения `{2048, 4096, 8192, 16384}`, но один
снимок состояния всегда использует ровно одну размерность. Смешивание кодеков
или размерностей в одном состоянии запрещено.

### 2.2 Детерминизм

- Все архитектурные мутации выполняет один `StateWriter`.
- `asyncio` используется для ввода-вывода и подготовки неизменяемых предложений,
  но не для конкурентной записи в состояние.
- Каждое предложение имеет `base_epoch`; устаревшее предложение отклоняется.
- Все tie-break выполняются по `(priority, created_tick, organ_id, signal_id)`.
- Случайные векторы вычисляются из `PRF(global_seed, namespace, stable_id)`.
- В нормативном пути запрещены недетерминированные GPU kernels.
- Время внутри перехода — логический `tick`, не wall-clock.
- Один config, event log и snapshot обязаны давать одинаковый `state_hash` и
  одинаковые выходы на одной нормативной CPU-реализации.

PyTorch разрешён только в экспериментальных heads. Их результаты не входят в
гарантию побитового replay, если отдельно не включён deterministic profile.

### 2.3 Каноническая сериализация и идентификаторы

- Состояние и события хешируются как UTF-8 JSON с NFC-строками, сортированными
  ключами, без insignificant whitespace и без floating-point значений.
- Q16 сериализуется как знаковое целое число, обозначающее значение `n/65536`.
- `signal_id = UUIDv5(run_namespace, causal_id || organ_id || ordinal)`.
- `record_id = first_128_bits(SHA256(canonical_record_identity))`; при
  коллизии добавляется детерминированный ordinal из лексикографического порядка
  полных хешей.
- `snapshot_id = SHA256(config_hash || epoch || canonical_state_root)`.
- Token ID назначается последовательно при первом появлении в event-log;
  одинаковый event-log поэтому создаёт один словарь. Внутри атомарного batch
  новые токены сортируются по нормализованной UTF-8 строке.
- Все timestamps внутри нормативного состояния равны logical tick. Wall-clock
  хранится только в ненормативных diagnostic metadata и не входит в state hash.
- Символьные n-граммы используют явные markers `⟨BOW⟩` и `⟨EOW⟩`; пустая
  строка, invalid UTF-8 и token длиннее лимита отклоняются до выдачи ID.

---

## 3. HDC-алгебра

### 3.1 Типы векторов

Используются два связанных представления:

```text
IdentityHV  := {-1,+1}^D        # неизменяемая идентичность
SemanticAcc := int16^D          # изменяемый накопитель смысла
SemanticHV  := sign(SemanticAcc)
```

**[РЕФЕРЕНС] Критическое разделение:** `IdentityHV` никогда не меняется.
Динамичность архитектуры реализуется через версионируемый `SemanticAcc`.
Связи и адреса памяти используют стабильные identity-векторы; similarity и
обучение могут использовать semantic-векторы. Поэтому локальное обучение не
перекодирует ретроактивно всю память.

### 3.2 Примитивы

Для биполярных векторов:

```text
bind(a, b)       = a ⊙ b
unbind(ab, a)    = ab ⊙ a
permute(a, p)    = cyclic_rotate(a, 137 * p mod D)
bundle(v_1..v_n) = sign(sum(v_i))
similarity(a, b) = dot(a, b) / D
```

При нулевой сумме координаты знак задаётся битом
`IdentityHV("tie", coordinate)`. Это делает bundle детерминированным.

Bind не объявляется универсально ортогональным операндам. Near-orthogonality
ожидается только для независимых случайных операндов и проверяется статистически.

### 3.3 Динамический семантический слой

Для понятия `x` хранятся:

```text
identity_x = PRF_HV(seed, "identity", stable_id_x)
acc_x      = int16[D]
semantic_x = sign(acc_x)
```

При создании `acc_x = identity_x * A0`, где `A0 = 8`.

Одно принятое наблюдение контекста `c` создаёт предложение:

```text
delta_x = clip_int8(round(weight * context_hv(c)))
acc'_x  = clip_int16(acc_x + eta_mode * delta_x + rho * identity_x)
```

где:

- `weight ∈ [-1,1]` хранится в Q16;
- `eta_mode ∈ {1,2,4}` для discovery/memorization/practice;
- `rho ∈ {0,1}` — слабое якорение идентичности;
- за один tick каждая координата меняется не более чем на `4`;
- обновление применяется copy-on-write внутри транзакции epoch.

Неизменяемая запись связывает `semantic_version` и `base_epoch`. Старые записи
не пересчитываются автоматически. Индекс semantic similarity обновляется только
после commit новой версии.

### 3.4 Инварианты динамики

1. `identity_x` побитово неизменяем.
2. Любая версия `semantic_x` имеет родителей и provenance.
3. Локальный update не мутирует записи, не входящие в declared dependency set.
4. За транзакцию cosine drift ограничен конфигурацией `max_semantic_drift=0.02`.
5. Превышающий лимит update дробится или отклоняется как `DRIFT_LIMIT`.
6. Rollback восстанавливает accumulator, semantic index и все маршруты одного
   epoch атомарно.

---

## 4. Кодирование входов

### 4.1 Текст

Текст нормализуется Unicode NFC и сегментируется детерминированным токенизатором.
Токен кодируется bundle символьных n-грамм длины 3–5 с boundary markers.
Учитываются первые 96 уникальных n-грамм в лексикографическом порядке.

Позиционный контекст длины `K ≤ 8`:

```text
context_hv = bundle(
  bind(role(1), permute(token_hv(t-K+1), 1)),
  ...,
  bind(role(K), permute(token_hv(t), K))
)
```

Роль использует неизменяемый `IdentityHV`. Порядок поэтому не зависит от
последовательности обновления semantic-векторов слов.

### 4.2 Тело

**[ИСТОЧНИК]** Поддерживаются четыре канала: visual, auditory, pressure,
temperature.

**[РЕФЕРЕНС]** В v3 это нормализованные скаляры `[0,1]`, а не полноценные
изображения или звук. Каждый канал квантуется в 256 уровней. Level-векторы
строятся так, чтобы ожидаемая similarity монотонно убывала с расстоянием уровней.

```text
BodyState {
  visual: Q16,
  auditory: Q16,
  pressure: Q16,
  temperature: Q16,
  valid_mask: uint4,
  observed_tick: uint64
}
```

Интервенция в канал обязана иметь измеримое предсказанное влияние на конкретный
орган или память; иначе канал считается декоративным и выключается ablation.

---

## 5. Полное состояние

```text
State {
  spec_version,
  config_hash,
  epoch: uint64,
  tick: uint64,
  rng_counter: uint64,
  body: BodyState,
  resources: ResourceState,
  hormones: HormoneState,
  emotions: EmotionState,
  coherence: CoherenceState,
  needs: NeedState[],
  wants: Want[],
  goals: Goal[],
  learning_modes: PatternModeTable,
  concepts: ConceptRegistry,
  stm: MemoryManifest,
  ltm: MemoryManifest,
  bus: SignalQueue,
  scheduler: SchedulerState,
  output: OutputState,
  event_log_offset: uint64,
  parent_snapshot: SnapshotId | null
}
```

Каждый tick формирует новый логический снимок. Реализация может структурно
разделять неизменившиеся страницы, но опубликованный `state_hash` охватывает все
перечисленные поля и активные индексы.

---

## 6. Сигнальная шина и органы

### 6.1 Сигнал

```text
Signal {
  signal_id: UUIDv5,
  causal_id: UUIDv5,
  parent_id: UUIDv5 | null,
  base_epoch: uint64,
  created_tick: uint64,
  kind: enum,
  payload_ref: ContentHash,
  hv: IdentityHV | SemanticHV | null,
  intensity: Q16,
  urgency: Q16,
  confidence: Q16,
  provenance: Provenance,
  visited_organs: BitSet,
  hop_count: uint8,
  deadline_tick: uint64
}
```

Сигнал неизменяем. Орган создаёт дочерний сигнал и `StateDeltaProposal`; он не
мутирует исходный сигнал или State напрямую.

### 6.2 Интерфейс органа

```text
Organ {
  organ_id: StableId,
  accepts(signal, state_view) -> bool,
  relevance(signal, state_view) -> Q16,
  propose(signal, state_view) -> Proposal,
  cost_bound(signal) -> ResourceCost,
  invariant_set -> InvariantId[]
}
```

`Proposal` содержит read-set, write-set, base_epoch, predicted resource cost,
дочерние сигналы, falsifier tag и обратимую delta либо явный `IRREVERSIBLE`.
Необратимые предложения запрещены в рабочем цикле и допустимы только после
отдельного maintenance commit с checkpoint.

### 6.3 Нормативный каталог органов

1. `PerceptionEncoder` — кодирует вход.
2. `NoveltyDetector` — сравнивает с STM/LTM.
3. `MemoryRetriever` — извлекает ассоциации.
4. `SequencePredictor` — формирует next-token/action prediction.
5. `PredictionError` — вычисляет наблюдаемую ошибку.
6. `NeedUpdater` — обновляет дефициты.
7. `WantGenerator` — строит кандидаты действий.
8. `GoalArbiter` — публикует/завершает цели.
9. `ActionSelector` — выбирает ограниченное действие.
10. `HormoneRegulator` — обновляет медленные регуляторы.
11. `EmotionDecoder` — вычисляет производные краткие режимы.
12. `LearningUpdater` — предлагает semantic update и режим обучения.
13. `MemoryWriter` — пишет STM.
14. `ConsolidationPlanner` — строит candidate STM/LTM commit.
15. `LoopBreaker` — реагирует на повторяющийся causal trace.
16. `OutputPublisher` — публикует проверенный выход.

### 6.4 Планировщик

Для допустимого `(organ, signal)` вычисляется:

```text
priority = clamp01(
    0.20 * organ.base_priority
  + 0.20 * relevance
  + 0.15 * signal.urgency
  + 0.15 * active_goal_alignment
  + 0.10 * need_pressure
  + 0.10 * queue_age
  + 0.10 * hormone_gain
  - 0.15 * normalized_cost
  - 0.20 * loop_penalty
)
```

Веса конфигурируемы, но их сумма и диапазоны валидируются. Hormone gain не может
умножить priority более чем на 2 или опустить ниже 0.5 эквивалента базового
значения. `queue_age` обеспечивает ограниченную защиту от starvation.

Один tick исполняет не более `max_events_per_tick=256`. Один causal chain имеет
не более `max_signal_hops=64`. Повтор пары `(organ_id, state_projection_hash)`
три раза создаёт `LOOP_DETECTED`; дальнейший dispatch этой пары блокируется до
следующего epoch.

### 6.5 Конфликты записей

StateWriter сортирует предложения детерминированно. Пересекающиеся write-set:

- коммутативные deltas объединяются зарегистрированным merge operator;
- некоммутативные предложения сериализуются и второе пересчитывается;
- если пересчёт невозможен, второе получает `STALE_PROPOSAL`.

Commit либо применяет всю delta и дочерние сигналы, либо ничего.

---

## 7. Ограниченные ресурсы

```text
ResourceState {
  energy: Q16,
  compute_tokens: uint32,
  queue_pressure: Q16,
  recovery_rate: Q16,
  sleep_debt: Q16
}
```

На organ call списывается объявленная верхняя оценка, после выполнения —
корректируется до фактической стоимости. Ресурс никогда не опускается ниже нуля.

```text
energy' = clamp01(energy - cost + recovery)
sleep_debt' = clamp01(sleep_debt + work_load - sleep_recovery)
```

Критические сигналы получают зарезервированные 10% compute budget. Некритическая
очередь не может вытеснить их полностью. При arrival rate выше service rate
включается backpressure: новые пользовательские входы получают `BUSY`, а не
накапливаются без границы.

Сон запускается при `(idle_ticks ≥ 128 OR sleep_debt ≥ 0.8)` и отсутствии
критического сигнала. Он ограничен 512 maintenance operations и прерываем.

---

## 8. Гормоны и эмоции

### 8.1 Гормоны

```text
HormoneState {
  dopamine: Q16,
  serotonin: Q16,
  cortisol: Q16,
  adrenaline: Q16,
  two_ag: Q16
}
```

Это инженерные каналы `[0,1]`. Для каждого `h` задаются baseline `b_h`, decay
`d_h` и bounded release `r_h(t)`:

```text
h(t+1) = clamp01(h(t) + r_h(t) - d_h * (h(t) - b_h))
```

Нормативные источники release:

| Канал | Наблюдаемый источник | Основное влияние |
|---|---|---|
| dopamine | положительная дельта prediction/task utility | reinforcement, goal persistence |
| serotonin | устойчивое завершение и рост coherence без потери utility | damping, consolidation guard |
| cortisol | prediction error, conflict, invariant violation | caution, quarantine, memory inhibition |
| adrenaline | deadline и критическая внешняя интенсивность | urgency, зарезервированный compute |
| two_ag | loop score и повтор causal trace | suppress repeated route, boost alternative |

Release каждой реакции ограничен `0.05` за tick. Один канал не может напрямую
переписать память: он только изменяет bounded коэффициенты admission/priority.

### 8.2 Устойчивость

- Все gains ограничены `[0.5, 2.0]`.
- Интеграторы имеют anti-windup.
- При нулевом входе гормоны обязаны вернуться в `±0.01` baseline за объявленное
  `settling_time`.
- Если наблюдается осцилляция амплитудой >0.1 в течение 32 ticks, controller
  переходит в `REGULATOR_SAFE_MODE`: gains=1, новые releases блокируются,
  событие журналируется.

### 8.3 Эмоции

Эмоция — производная, а не независимый скрытый state:

```text
emotion_k_activation = clamp01((w_k · hormones - enter_k) / width_k)
```

Для выхода используется `exit_k < enter_k` (hysteresis). Пересекающиеся эмоции
разрешены; output — вектор активаций, а не одно взаимоисключающее имя.
Эмоция может добавить к priority не более `±0.10`. Без отдельной причинной
абляции эмоциональный слой считается визуализацией гормонального состояния.

---

## 9. Когерентность

Когерентность не является качеством ответа и не заменяет внешнюю task utility.
Это диагностическая оценка внутренней согласованности:

```text
coherence = clamp01(1 - (
    0.20 * invariant_violation_rate
  + 0.15 * normalized_prediction_error
  + 0.15 * unresolved_conflict_rate
  + 0.15 * loop_score
  + 0.10 * queue_pressure
  + 0.10 * regulator_instability
  + 0.10 * memory_integrity_risk
  + 0.05 * state_range_violation
))
```

Все составляющие имеют независимые observables и окна измерения 32 ticks.

Правила против Goodhart:

1. Бездействие не считается успехом: task utility публикуется отдельно.
2. Stabilization не может удалить, слить или скрыть externally verified record.
3. Решение оптимизируется лексикографически: safety invariants → task utility →
   coherence → resource cost.
4. Повышение coherence при падении task utility более чем на 5% считается
   `COHERENCE_REGRESSION`.
5. Формула и нормы фиксируются до закрытого теста.

---

## 10. Потребности, желания и цели

### 10.1 Потребности

Need — непрерывный дефицит `[0,1]`, а не правило действия:

```text
Need {
  need_id,
  deficit: Q16,
  target_range,
  weight: Q16,
  evidence_refs[],
  updated_tick
}
```

В v3 определены needs: energy, safety/coherence, task commitment, novelty,
memory integrity. Body-derived needs включаются только в экспериментах тела.

### 10.2 Желания

Want — оценённый кандидат действия:

```text
want_score(action) =
  sum_i need_i.weight * predicted_deficit_reduction_i
  + expected_task_utility
  + goal_compatibility
  - resource_cost
  - predicted_risk
```

Все terms находятся в Q16 `[0,1]`. Предсказания с confidence ниже 0.5 получают
штраф неопределённости. Want не выполняет действие самостоятельно.

### 10.3 Цели

Want становится Goal, если score ≥0.75 не менее трёх ticks либо ≥0.9 один tick.

```text
Goal {
  goal_id,
  source_want,
  success_predicate,
  failure_predicate,
  deadline_tick,
  resource_budget,
  priority,
  status,
  provenance
}
```

Одновременно активны не более восьми целей. Конфликт разрешается сначала safety,
затем deadline, priority, created_tick и goal_id. Осцилляция двух целей более
четырёх переключений создаёт meta-goal `RESOLVE_CONFLICT` либо завершает менее
приоритетную цель как `BLOCKED`.

---

## 11. Режимы обучения

Для каждого stable pattern ID хранится:

```text
PatternStats {
  exposures,
  successful_uses,
  failed_uses,
  last_seen,
  novelty,
  confidence,
  mode: DISCOVERY | MEMORIZATION | PRACTICE
}
```

Переходы:

- `DISCOVERY`: exposures < 3 или confidence < 0.45;
- `MEMORIZATION`: exposures ≥ 3 и confidence ≥ 0.45;
- `PRACTICE`: exposures ≥ 8, successful_uses ≥ 3 и confidence ≥ 0.70;
- PRACTICE → MEMORIZATION после двух ошибок в окне пяти применений;
- MEMORIZATION → DISCOVERY при confidence <0.35 или обнаруженном конфликте.

Повторение ложной строки не является успехом: `successful_use` требует внешнего
oracle, предсказания следующего наблюдения или выполнения goal predicate.
Exact repetition и paraphrased/compositional application публикуются отдельно.

Learning rate semantic-вектора зависит от режима, но остаётся в пределах §3.3.

---

## 12. STM, LTM и запись памяти

### 12.1 Запись

```text
MemoryRecord {
  record_id,
  type,
  identity_hv,
  semantic_version,
  payload_hash,
  provenance,
  source_trust,
  created_tick,
  last_access_tick,
  frequency,
  utility,
  salience,
  confidence,
  parents[],
  status: ACTIVE | QUARANTINED | TOMBSTONED,
  store: STM | LTM,
  epoch
}
```

Каждый новый опыт сначала записывается в STM. Прямая запись в LTM разрешена
только для externally verified record с trust ≥0.95 и всё равно проходит
транзакционный путь promotion.

### 12.2 Salience и влияние гормонов

```text
salience = clamp01(
    0.25 * novelty
  + 0.20 * goal_relevance
  + 0.20 * prediction_error_information
  + 0.15 * repetition_support
  + 0.10 * source_trust
  + 0.10 * bounded_hormone_memory_gain
)
```

Гормоны меняют итог не более чем на `±0.10`; они не могут превратить untrusted
record в trusted. Cortisol повышает quarantine threshold, dopamine повышает
reinforcement только после подтверждённого success, serotonin увеличивает
минимальное witness coverage для merge.

### 12.3 Вместимость

Фраза «память не ограничена» трактуется как отсутствие архитектурно заданного
человеческого лимита, но любая реализация обязана объявить конечные capacity.

- STM: 4,096 active records;
- LTM: 65,536 active records;
- версии: не более 2x active capacity;
- event log: 1,000,000 событий между архивными checkpoints;
- при заполнении — `CAPACITY_PRESSURE`, затем consolidation или backpressure;
- скрытое удаление для освобождения места запрещено.

### 12.4 Promotion STM → LTM

Promotion допускается при:

```text
(frequency >= 3 OR successful_uses >= 1)
AND confidence >= 0.60
AND source_trust >= 0.50
AND status == ACTIVE
AND no unresolved contradiction
```

Протокол exactly-once:

1. зафиксировать hormone sample и source epoch;
2. создать LTM candidate с тем же logical `record_id` и новой version;
3. построить candidate индексы;
4. записать WAL и `fsync`;
5. CAS публикует общий memory root `(STM, LTM, indices, epoch)`;
6. старая STM version становится tombstone в том же commit;
7. crash до CAS оставляет STM активной; crash после CAS восстанавливает LTM.

Запись никогда не может быть одновременно активна в двух stores.

---

## 13. Консолидация и сон

### 13.1 Кандидаты merge

Две записи могут слиться только если:

- одинаковы `type` и schema version;
- similarity ≥0.94;
- нет отрицания, несовместимого числа, времени, scope или entity ID;
- outcome distributions имеют normalized L1 distance ≤0.10;
- provenance обоих родителей сохраняется;
- все mandatory witnesses дают тот же результат до и после;
- merge уменьшает physical cost.

Кандидаты сортируются по
`(-similarity, min(record_id), max(record_id))`; за sleep-цикл выполняется не
более 32 merge. Пересекающиеся пары разрешаются greedy в этом порядке.

Merged semantic accumulator — покоординатная насыщаемая сумма родителей с
весами support. Новый record получает content-derived ID, список родителей,
суммированные counters с проверкой overflow и union свидетелей до лимита 32.

### 13.2 Удаление

Немедленное физическое удаление запрещено. Этапы:

1. `ACTIVE → QUARANTINED` с причиной;
2. минимум два snapshot epochs и 1,024 ticks наблюдения;
3. blind retention/dependency probes;
4. `QUARANTINED → TOMBSTONED` обратимым commit;
5. физический GC только после истечения rollback window и архивного checkpoint.

Externally verified, rare-critical и goal-dependent записи автоматически не
удаляются. Ошибка удаления откатывается сменой memory root.

### 13.3 Публикация

Consolidation строит candidate root copy-on-write. Admission witnesses отделены
от закрытого evaluation set. WAL → fsync → CAS публикует новый epoch. In-flight
query pinned к старому epoch может вычисляться, но его ответ после CAS не
публикуется и заменяется `RETRY`.

Сон не имеет права изменять hormone baseline, needs или goal success predicates.
Он прерывается критическим сигналом после завершения текущей атомарной операции.

---

## 14. Главный цикл

Нормативная функция является тотальной:

```text
step(state, external_event) -> (new_state, outputs, trace)
```

1. Валидировать schema, размер и provenance события.
2. При ошибке вернуть `INVALID_INPUT` без мутации.
3. Закодировать вход и создать root signal.
4. Пока очередь не пуста и budget не исчерпан:
   - выбрать `(organ, signal)` по §6.4;
   - проверить hop/deadline/resource/invariants;
   - получить proposal на immutable state view;
   - проверить read/write set и base_epoch;
   - применить атомарный commit либо явный status;
   - добавить дочерние сигналы.
5. Обновить hormones, emotions, coherence и resources один раз на logical tick.
6. Проверить goals и сформировать bounded action/output candidates.
7. OutputPublisher повторно проверяет active epoch и safety invariants.
8. Записать event, trace, config hash и state hash.
9. Вернуть опубликованный snapshot.

Все ветви имеют исход: `COMMITTED`, `NO_OP`, `INVALID_INPUT`, `BUSY`,
`CAPACITY_PRESSURE`, `DRIFT_LIMIT`, `STALE_PROPOSAL`, `LOOP_DETECTED`,
`RESOURCE_EXHAUSTED`, `RETRY`, `SAFE_MODE` или `INVARIANT_FAILURE`.

---

## 15. Выход и практическое использование

В текстовом v3 OutputHead выполняет только:

1. retrieval ассоциированного токена/паттерна;
2. next-token prediction;
3. продолжение не более 32 токенов с остановкой по UNKNOWN/loop/deadline;
4. выбор одного зарегистрированного действия из конечного ActionRegistry.

Каждый результат содержит confidence, memory provenance, active goal, epoch и
trace ID. Свободная генерация произвольного текста без provenance не считается
доказательством обучения.

---

## 16. Экспериментальная программа

### 16.1 Unit-тесты HDC

- распределение similarity независимых, коррелированных и self-bind векторов;
- bind/unbind recovery;
- member/non-member AUC bundle против числа элементов и D;
- различение перестановок порядка;
- drift и retention динамического semantic overlay;
- побитовый replay одной транзакции.

### 16.2 Пушкин

Эксперимент маркируется как demonstration of associative memorization. Метрики:
точность retrieval пропущенного слова, continuation знакомого фрагмента,
устойчивость к опечаткам и shuffled-order control. Он не используется для
утверждения об изучении русского языка.

### 16.3 Alice / online language experiment

Корпус делится до запуска:

- train chapters;
- held-out chapters;
- новые предложения с теми же конструкциями;
- role-reversal minimal pairs;
- agreement и word-order probes;
- nonce-token compositional probes;
- frequency-matched shuffled corpus.

Baselines при одинаковом memory и compute budget:

1. unigram frequency;
2. bounded n-gram table;
3. nearest-neighbor character n-gram retrieval;
4. static HDC;
5. dynamic semantic HDC;
6. HDC без bind;
7. HDC без permutation;
8. HDC без hormones/emotions;
9. единое хранилище вместо STM/LTM.

HDC-specific advantage заявляется только при улучшении закрытого test score
минимум на 5 процентных пунктов против лучшего equal-budget non-HDC baseline и
при отсутствии статистически значимого ухудшения retention.

### 16.4 Метрики

- held-out top-1/top-k accuracy;
- negative log rank;
- UNKNOWN precision/recall;
- retention после conflicting stream A→B;
- transfer на новые композиции;
- loop incidence и median/p99 loop length;
- task utility отдельно от coherence;
- STM/LTM promotion precision;
- false merge/delete rate;
- bytes, operations, p50/p95/p99 latency;
- exact replay и rollback equality.

Evaluation seeds, generators, sample sizes, hardware и statistical tests
пререгистрируются. Admission canaries не переиспользуются как итоговый test.

### 16.5 Референсная конфигурация бенчмарка

Если отдельная preregistration отсутствует, обязательны следующие значения:

- 20 независимых generator seeds;
- 10,000 train sequences и 2,000 test sequences на seed для synthetic grammar;
- не менее 1,000 probes каждого класса: retention, role reversal, agreement,
  word order, nonce composition, false merge и UNKNOWN;
- conflicting stream A и B содержат по 25,000 событий;
- saturation измеряется для `D ∈ {2048,4096,8192,16384}` и
  `N ∈ {1,2,4,...,16D}`;
- latency измеряется после 100 warm-up и на 10,000 queries, отдельно cold/warm;
- нормативное железо для отчёта: один CPU worker, не менее 8 GiB RAM, GPU off;
  точная модель CPU, OS, Python, NumPy и commit hash публикуются;
- основной интервал — paired bootstrap 95% CI на 10,000 resamples;
- множественные сравнения корректируются Holm method при family-wise α=0.05;
- публикуются все seeds, включая неудачные, median, mean, standard deviation,
  confidence interval и полный denominator отказов.

Пороговые gates:

- преимущество HDC над лучшим равнобюджетным non-HDC baseline: ≥5 п.п. top-1,
  нижняя граница paired 95% CI >0;
- retention drop dynamic против static HDC: ≤2 п.п.;
- обязательные false merge/delete probes: 0 нарушений;
- replay и retained-snapshot rollback: 100% побитовое совпадение;
- loop breaker: median loop length ниже control минимум на 50% при падении task
  success не более 2 п.п.;
- STM→LTM promotion precision ≥95% и recall ≥80%;
- память ни разу не превышает объявленный physical budget;
- любой отсутствующий denominator, seed или baseline делает результат INVALID,
  а не частично успешным.

---

## 17. Обязательные kill-tests

1. **Dynamic drift:** обновить один shared semantic vector и проверить все
   несвязанные старые ассоциации.
2. **Async schedule:** один log под разными допустимыми coroutine schedules.
3. **Bundle saturation:** `N=1…100D`, balanced и frequency-skewed inputs.
4. **Interference:** выучить A, затем конфликтующий B с frozen A probes.
5. **False merge:** likes/dislikes, safe/unsafe, 10/100, homonyms, temporal facts.
6. **Delayed utility:** редкая запись становится важной только после sleep.
7. **Routing loop:** два органа повторно активируют друг друга.
8. **Starvation:** дешёвый flood плюс редкий critical sentinel.
9. **Hormone impulse:** единичный stimulus, затем нулевой input до settling time.
10. **Coherence Goodhart:** стабильное ошибочное состояние и редкая истинная
    коррекция, временно понижающая coherence.
11. **STM/LTM race:** hormone threshold, write, promotion, crash, restart.
12. **Capacity:** уникальные vectors до лимита и ещё одна запись.
13. **Crash boundaries:** pre-WAL, WAL, fsync, CAS, index publish, GC.
14. **Live rollback:** задержанный output от child epoch после возврата parent.
15. **Goal oscillation:** две равные конфликтующие цели.
16. **Body ablation:** каналы on/off при одинаковом текстовом входе.
17. **Emotion ablation:** generic gain против hormone-only и hormone+emotion.
18. **Ninth outcome:** больше лимита исходов одного контекста должно иметь
    явный deterministic статус, а не зависеть от реализации.

---

## 18. Обработка переполнений и полные переходы

- Vocabulary full → `VOCABULARY_FULL`, событие не принимается.
- Outcome table full → deterministic Space-Saving: заменить минимальный count,
  tie-break максимальный token ID; replacement получает `min_count + 1`.
- Counter at `uint32 max` → атомарно разделить все counters записи на два с
  округлением вниз, минимум 1 для ненулевого значения.
- STM/LTM full → `CAPACITY_PRESSURE`; никакого скрытого eviction.
- Version pool full → сначала архивный checkpoint; при невозможности
  `VERSION_CAPACITY_FULL`.
- Event log full → snapshot + content-addressed archive; без подтверждённого
  архива вход получает `LOG_CAPACITY_FULL`.
- Partial batch error → весь batch откатывается, если не задан `atomic=false`.

Эти правила входят в state hash и одинаковы для всех реализаций.

---

## 19. Хранение, журнал и rollback

Snapshot охватывает HDC accumulators, STM/LTM manifests, semantic indices,
hormones, emotions, needs, goals, resources, queue, scheduler, RNG counter и
config hash.

WAL хранит:

```text
transaction_id, base_epoch, read_set_hash, delta_hash,
new_pages[], new_indices[], outputs[], commit_marker
```

Поддерживаются три опубликованных snapshot: current, parent и grandparent.
Rollback меняет весь root, а не только память. После rollback epoch увеличивается,
чтобы in-flight child outputs не могли пройти publication barrier.

Архивный журнал может быть вынесен на диск, но входит в полный memory/storage
accounting эксперимента. GC удаляет только страницы, недостижимые из retained
snapshots и подтверждённого archive root.

---

## 20. P2P и стигмергия — экспериментальное расширение

Тысячи агентов не обмениваются raw hypervectors: разные seeds создают
несовместимые basis. Межузловой формат символический и content-addressed:

```text
StigmergicTrace {
  trace_id,
  author_agent,
  schema_id,
  symbolic_payload,
  provenance,
  confidence,
  utility_evidence,
  ttl,
  parent_traces[],
  signature
}
```

Получатель локально кодирует trace в свой HDC basis. Протокол требует dedup по
trace ID, TTL, rate limits, trust policy, quarantine, loop detection и partition
tests. Конфликтующие traces не разрешаются количеством голосов и не входят в LTM
без локального verification event.

P2P не входит в acceptance v3 single-agent core. Он допускается только после
прохождения core benchmark и отдельного теста 1,000 peers на reorder,
duplication, partitions, Sybil amplification, convergence и traffic bounds.

---

## 21. Acceptance gates

Архитектура считается реализованной, только если:

1. 100 replay одного log дают одинаковый state hash;
2. ни один kill-test не нарушает safety invariants;
3. rollback восстанавливает полное поведение retained snapshot;
4. memory и operations остаются в объявленных границах;
5. consolidation не ухудшает mandatory witnesses;
6. loop breaker снижает loop duration без эквивалентного падения task success;
7. coherence предсказывает internal failures, но не заменяет task utility;
8. dynamic HDC не ухудшает retention против static HDC сверх 2 п.п.;
9. все биологические слои проходят причинные ablations;
10. заявления о языке делаются только при переносе на закрытые новые конструкции
    и превосходстве равнобюджетных baselines.

Непрохождение gate публикуется как `REJECTED_HYPOTHESIS`, а не маскируется
изменением метрики после эксперимента.

---

## 22. Соответствие исходной реконструкции

| Исходная идея | Реализация v3 | Статус происхождения |
|---|---|---|
| HDC bind/bundle/permute | §3 | источник + референсные параметры |
| динамические векторы | immutable identity + mutable semantic overlay | референсная безопасная интерпретация |
| четыре body channels | scalar level-HV channels | интерпретация |
| needs → wants → goals | continuous deficits and bounded arbitration | референс |
| state tuple | versioned total State | источник + референс |
| органы и кровеносная шина | immutable signals + reducers + deterministic scheduler | интерпретация + референс |
| ограниченные ресурсы | bounded energy/compute/backpressure | референс |
| гормоны и 2-АГ | bounded slow controllers | источник + референсные equations |
| эмоции | derived hysteretic regions | интерпретация |
| когерентность | diagnostic internal-consistency score | интерпретация + anti-Goodhart rules |
| STM/LTM | transactional bounded stores | источник + референс |
| сон | bounded preemptible maintenance | интерпретация |
| три режима обучения | explicit transition table | источник + референс thresholds |
| Пушкин/Alice | controlled memorization/language experiments | исправленный protocol |
| P2P stigmergy | symbolic signed traces, not raw HDC | референсное расширение |

---

## 23. Остаточные ограничения

- Пороговые значения являются стартовой гипотезой и требуют preregistered sweep.
- Dynamic semantic overlays могут всё ещё создавать interference.
- HDC capacity конечна и зависит от D, корреляции и frequency skew.
- Typed guards не обнаруживают все семантические противоречия.
- Coherence остаётся инженерной диагностикой, не свойством сознания.
- Скалярное «тело» не является embodied cognition.
- Ассоциативное продолжение текста не равно пониманию языка.
- P2P расширение требует отдельной модели угроз и консенсуса.
- Полная биологическая правдоподобность не является целью этой спецификации.

Главная ценность v3 — не обещание когнитивного прорыва, а возможность двум
независимым командам реализовать одну и ту же архитектуру, получить сравнимые
трассы и экспериментально отвергнуть неработающие части.
