# 🚀 Инструкция по Push в GitHub

## 1. Создать Personal Access Token

Перейти: https://github.com/settings/tokens

- Нажать **"Generate new token (classic)"**
- Name: `dots-ocr-ultimate`
- Expiration: `90 days` (или No expiration)
- Отметить: ✅ **repo** (все галочки)
- Нажать **"Generate token"**
- **СКОПИРОВАТЬ токен** (показывается один раз!)

## 2. Запушить в GitHub

### Вариант А: С сохранением токена в remote

```bash
cd /srv/dots.ocr/dots.ocr-ultimate

# Настроить remote (замените YOUR_TOKEN)
git remote set-url origin https://YOUR_TOKEN@github.com/ansidenko/dots.ocr-ultimate.git

# Пуш master
git push -u origin master

# Пуш ветки с интеграциями
git push -u origin ultimate-integration
```

### Вариант Б: Без сохранения токена (ввод при каждом пуше)

```bash
cd /srv/dots.ocr/dots.ocr-ultimate

# Просто пуш
git push -u origin master

# Когда спросит:
# Username: ansidenko
# Password: ваш_токен_вместо_пароля
```

## 3. После пуша

Сказать: **"запушил"** 

И продолжим интеграцию компонентов! 🎯

