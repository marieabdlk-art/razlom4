# Blind benchmark harness

Это offline scorer для сравнения пяти режимов рассуждения. Текущий банк из
трёх задач — только smoke-test механики и не является результатом оценки
РАЗЛОМ-4.

## Граница слепоты

Модельному runner разрешено читать только `public-task-bank.json` и
`manifest.json`. `private-task-bank.json` должен находиться в отдельном
scorer-процессе или закрытом evaluation workspace. Простое указание модели «не
смотри в файл» не считается слепым тестом.

## Что реализовано

- semantic validation public/private bank;
- canonical SHA-256 seal;
- одинаковый total-token ceiling для всех arms;
- отдельные call caps согласно протоколу режима;
- fail-closed validation полного submission;
- deterministic hidden-oracle scoring;
- exact success, false solution, coverage, probe accuracy, failure recall;
- Wilson 95% interval и paired exact sign-test;
- запрет содержательного claim при числе задач меньше 40.

Полный протокол и критерии находятся в `../RAZLOM4_BENCHMARK_PLAN.md`.

## Provider boundary

Этот каталог намеренно не содержит API-ключей и автоматического live-runner.
Provider runner должен получать публичную задачу, arm prompt и token cap,
возвращать submission JSON и usage. Только после завершения всех arms закрытый
scorer получает submissions и private bank.
