" Dein.vim
if has ('nvim')
    " Set dein paths
    let s:config_home = expand('~/.config')
    let s:nvimdir = s:config_home . '/nvim'
    let s:cache_home = expand('~/.cache')
    let s:dein_dir = s:cache_home . '/dein'
    let s:dein_github = s:dein_dir . '/repos/github.com'
    let s:dein_repo_name = 'Shougo/dein.vim'
    let s:dein_repo_dir = s:dein_github . '/' . s:dein_repo_name
    " Check dein has been installed or not.
    if !isdirectory(s:dein_repo_dir)
        let s:git = system("which git")
        if strlen(s:git) != 0
            echo 'dein is not installed, install now.'
            let s:dein_repo = 'https://github.com' . '/' .  s:dein_repo_name
            echo 'git clone ' . s:dein_repo . ' ' . s:dein_repo_dir
            call system('git clone ' . s:dein_repo . ' ' . s:dein_repo_dir)
        endif
    endif
    " Add the dein installation directory into runtimepath
    let &runtimepath = &runtimepath . ',' . s:dein_repo_dir
    " Begin plugin installation
    if dein#load_state(s:dein_dir)
        call dein#begin(s:dein_dir)
            let s:toml = s:nvimdir . '/dein.toml'
            let s:lazy_toml = s:nvimdir . '/dein_lazy.toml'
            call dein#load_toml(s:toml, {'lazy': 0})
            call dein#load_toml(s:lazy_toml, {'lazy': 1})
        call dein#end()
        call dein#save_state()
    endif
    " Installation check
    if dein#check_install()
        call dein#install()
    endif
endif

" ----- visual/ -----
set number
set noerrorbells
set showmatch matchtime=1
set cinoptions+=:0
set cmdheight=2
set showcmd
set display=lastline
set list
set showmatch
set cursorline
set autoread
" ----- /visual -----


" ----- search/ -----
set ignorecase
set smartcase
set wrapscan
set incsearch
set hlsearch
" ----- /search -----


" ----- edit/ -----
inoremap jj <Esc>
noremap <Esc><Esc> :noh<CR>
noremap ; :
set clipboard+=unnamedplus
set expandtab
set shiftwidth=2
set tabstop=2
set softtabstop=2
set smartindent
" ----- /edit -----


" ----- control/ ----
nmap <C-h> <C-w>h
nmap <C-j> <C-w>j
nmap <C-k> <C-w>k
nmap <C-l> <C-w>l
" ----- /control ----


" ----- other/ -----
set ttimeoutlen=10
" ----- /other -----


" Folding
set foldmethod=marker
set foldlevel=0
set foldcolumn=3

" relative line numbers
aug numbertoggle
    au!
    au BufEnter,FocusGained,InsertLeave *
        \ set relativenumber
    au BufLeave,FocusLost,InsertEnter   *
        \ set norelativenumber
  aug END