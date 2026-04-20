# CONTRIBUTING

Queste sono le regole del repository per evitare confusione e commit diretti su branch protetti.

## Branch

- MAI toccare il `main`. Usare `dev` come linea principale.
- All'inizio di ogni settimana creare un branch dalla `dev` chiamato `sprint00`, `sprint01`, ...
- Da un branch `sprint0x` creare branch per singole feature (es: `feature/data-preprocessing`, `feature/canonical-smiles-column`).
- Nei branch feature lavora preferibilmente UNA SOLA persona alla volta.
- Quando la feature è pronta: commit + push sul branch feature, aprire PR e merge nel `sprint0x`.
- Ogni fine settimana: merge del branch `sprint0x` nel branch `dev`.
- MAI committare direttamente in: `main`, `dev`, `sprint*`. Commmitare solo nei branch dedicati alle feature.

Questa divisione in sprint serve a separare i lavori per settimana.

## Commit

- Messaggi di commit significativi: descrivete cosa avete fatto e, se utile, cosa non siete riusciti a fare.

## Comandi consigliati (ordine)

1) Creare lo sprint settimanale (da usare all'inizio della settimana):

```bash
git checkout dev
git pull origin dev
git checkout -b sprint01
git push -u origin sprint01
```

2) Creare una branch feature da uno sprint:

```bash
git checkout sprint01
git pull origin sprint01
git checkout -b feature/nome-feature
# Lavori locali...
git add .
git commit -m "Breve descrizione: cosa/why"
git push -u origin feature/nome-feature
```

3) Quando la feature è pronta, aprire PR verso `sprint01` (o fare merge locale se concordato):

```bash
# (opzionale) merge locale
git checkout sprint01
git pull origin sprint01
git merge --no-ff feature/nome-feature
git push origin sprint01
```

4) Fine settimana: merge dello sprint in `dev` (da fare via PR o locally):

```bash
git checkout dev
git pull origin dev
git merge --no-ff sprint01
git push origin dev
```

5) Note su `main`:

- `main` è il ramo stabile/di rilascio. Non fare mai commit diretto su `main` dal workspace: usate PR autorizzate e una politica di merge condivisa.

## Suggerimenti rapidi

- Usate PR per revisioni e per mantenere la storia pulita.
- Se siete in dubbio, chiedete sul canale di progetto prima di effettuare merge importanti.

---
Se volete, posso creare i branch `sprint01` e i file `slides_9-11.md` ecc. nel repo ora.
