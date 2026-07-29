# РАЗЛОМ-4: preregistered blind benchmark

## Исследовательский вопрос

При одинаковом полном модельном бюджете повышает ли полный протокол
`РАЗЛОМ-4` точность причинного решения и обнаружение критических дефектов по
сравнению с более простыми способами расходовать те же токены?

Benchmark оценивает самостоятельный метод рассуждения, а не конкретного
провайдера моделей.

## Пять режимов

1. `single_agent` — один агент формирует финальное решение.
2. `best_of_4` — четыре слепых предложения и отдельный выбор.
3. `ordinary_debate` — четыре предложения, открытая дискуссия и выбор судьёй.
4. `razlom4_no_roles` — три фазы РАЗЛОМ-4, но все участники получают одинаковую
   функцию качества.
5. `razlom4_full` — четыре несовместимые роли, слепой commit, разлом, мутация,
   анонимная гильотина и детерминированный отбор.

Все режимы получают одинаковый `total_token_budget_per_task`. Разное число
вызовов допустимо, но превышение общего бюджета делает результат задачи
невалидным. Температура, модель, provider и контекст задачи должны совпадать.

## Слепота

- `benchmark/public-task-bank.json` можно передавать моделям.
- `benchmark/private-task-bank.json` доступен только scorer-процессу.
- `seal` фиксирует SHA-256 обоих банков и preregistration до первого запуска.
- Порядок режимов внутри каждой задачи должен определяться внешним runner по
  заранее записанному seed.
- Scorer не использует LLM и не видит рассуждения или авторство режима.
- После seal нельзя изменять task bank; исправление создаёт новый benchmark id.

## Задачи и ответ

Задача публикует контракт, наблюдения, набор допустимых причинных операторов,
assumptions и probes. Режим возвращает закрытый JSON:

- `SOLUTION` или `NO_VALID_SOLUTION`;
- выбранные operator ids;
- отброшенные assumption ids;
- прогнозы исходов опубликованных probes;
- найденные failure ids;
- каноническую карточку механизма;
- фактическое число calls и токенов.

Private oracle допускает несколько эквивалентных валидных решений. Exact
success требует совпадения с одним полным oracle-вариантом. `SOLUTION`, не
прошедший oracle, считается `false_solution`; отказ не переименовывается в
успех.

## Primary endpoints

1. `exact_success_rate` — основной показатель.
2. `false_solution_rate` — safety endpoint; рост недопустим.
3. `probe_accuracy` — точность различающих прогнозов.
4. `failure_recall` — доля найденных скрытых критических дефектов.
5. `coverage` — доля задач, где режим заявил `SOLUTION`.
6. Реальные calls/input/output tokens.

Primary comparison: `razlom4_full` против каждого из четырёх controls на одних
и тех же задачах. Для exact success публикуется точный двусторонний sign-test
по discordant pairs; дополнительно публикуются Wilson 95% intervals. Никакой
режим не исключается из denominator из-за `BLOCK`, ошибки JSON или бюджета.

## Размер исследования и критерий продолжения

В репозитории находится только трёхзадачный smoke bank, проверяющий harness. Он
не оценивает качество метода.

Перед содержательным прогоном нужно независимо создать и запечатать не менее
40 новых задач минимум из четырёх strata. Рекомендуемый минимум для заявления
о ценности:

- `razlom4_full` превосходит лучший control минимум на 10 процентных пунктов
  exact success;
- paired sign-test против лучшего control имеет `p < 0.05`;
- false solution не выше лучшего control;
- преимущество сохраняется минимум в трёх strata;
- evaluator не менялся после seal;
- результат повторён на второй модели или независимом task bank.

Если выигрыш исчезает при равном токен-бюджете или его даёт `best_of_4`,
специфическая ценность РАЗЛОМ-4 не подтверждена.

## Запуск offline scorer

```bash
python3 razlom4_benchmark.py validate-bank \
  benchmark/public-task-bank.json benchmark/private-task-bank.json
python3 razlom4_benchmark.py seal \
  benchmark/public-task-bank.json benchmark/private-task-bank.json \
  benchmark/manifest.json
python3 razlom4_benchmark.py evaluate \
  benchmark/public-task-bank.json benchmark/private-task-bank.json \
  benchmark/manifest.json benchmark/examples/smoke-submission.json
python3 razlom4_benchmark.py compare \
  benchmark/public-task-bank.json benchmark/private-task-bank.json \
  benchmark/manifest.json benchmark/examples/submissions
```

Live/API runner намеренно не включён в offline scorer. Его provider adapter
должен отдельно записывать сырые ответы и usage, не иметь доступа к private
bank и не запускаться без явно утверждённого бюджета.
