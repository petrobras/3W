# [ENGLISH]
# Git Large File Storage

## **First, why use and implement this functionality in this project?**

Repositories with files larger than 100 MiB are not automatically versioned by Git, as stated in its [documentation](https://docs.github.com/pt/repositories/working-with-files/managing-large-files/about-large-files-on-github). To do this, it is necessary to use Git LFS (Large File Storage).

## What does it do?
By indicating which files are larger than 100 MiB, Git LFS compresses them in order to make them lighter.

## Why use Git LFS?
Considering that data can take up a lot of space, this can be a strategy to avoid future headaches when trying to upload large files.

## How to use Git LFS?

1 - Install Git LFS according to your operating system. See this [link](https://git-lfs.com/) for more information.

2 - you need to indicate which file(s) should be compressed. To do this, use the command ``git track <filename>``.

For example 
(for files):
```git track "my_video.mp4"```

(for folders and all their files):
```git track "/example_folder/*"```

3 - After that, just continue normally with the commit, push, and other commands. 

[Documentation link](https://git-lfs.com/)


# [PORTUGUESE]

# Git Large File Storage

## **Primeiro, por que usar e implementar essa funcionalidade neste projeto?**

Repositórios com arquivos maiores que 100 MiB não são versionados automaticamente pelo Git, como consta em sua [documentação](https://docs.github.com/pt/repositories/working-with-files/managing-large-files/about-large-files-on-github). Para isso, é necessário usar o Git LFS (Large File Storage).

## O que ele faz?
Ao indicar quais arquivos são maiores que 100 MiB, o Git LFS os compacta para torná-los mais leves.

## Por que usar o Git LFS?
Considerando que os dados podem ocupar muito espaço, essa pode ser uma estratégia para evitar futuras dores de cabeça ao tentar fazer upload de arquivos grandes.

## Como usar o Git LFS?

1 - Instale o Git LFS de acordo com seu sistema operacional. Veja este [link](https://git-lfs.com/) para mais informações.

2 - você precisa indicar qual(is) arquivo(s) deve(m) ser compactado(s). Para fazer isso, use o comando ``git track <filename>``.

Por exemplo
(para arquivos):
```git track "my_video.mp4"```

(para pastas e todos os seus arquivos):
```git track "/example_folder/*"```

3 - Depois disso, basta continuar normalmente com os comandos commit, push e outros.

[Link da documentação](https://git-lfs.com/)