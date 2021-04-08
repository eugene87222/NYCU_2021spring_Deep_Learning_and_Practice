# -*- coding: utf-8 -*-
import re
import matplotlib.pyplot as plt


def complete_record(start_idx, stat):
    before = [0] * ((start_idx-1000)//1000)
    after = [stat[-1]] * (300-(start_idx-1000)//1000-len(stat))
    return before + stat + after


def parse_log(log_path):
    statistic = []
    details = []
    with open(log_path, 'r', encoding='utf-8') as fp:
        contents = fp.readlines()
        line_idx = 0
        while line_idx < len(contents):
            res = re.findall(r'\d+\s+mean = [\d|\.]+\s+max = \d+', contents[line_idx], re.S)
            if len(res):
                statistic.append(contents[line_idx])
                line_idx += 1
                detail = []
                while True:
                    if line_idx<len(contents) and contents[line_idx].strip().endswith(')'):
                        detail.append(contents[line_idx])
                        line_idx += 1
                    else:
                        details.append(detail)
                        break
            else:
                line_idx += 1
    iter_num = []
    mean = []
    maximum = []
    start_idx = {}
    win_rate = {}
    term_rate = {}
    for (stat, d) in zip(statistic, details):
        res = re.findall(r'(\d+)\s+mean = (.+)\s+max = (.+)', stat.strip(), re.S)[0]
        iter_num.append(int(res[0]))
        mean.append(float(res[1]))
        maximum.append(float(res[2]))
        for l in d:
            res = re.findall(r'(\d+)\s+(.+)%\s+\((.+)%\)', l.strip(), re.S)[0]
            if res[0] not in start_idx:
                start_idx[res[0]] = iter_num[-1]
                win_rate[res[0]] = []
                term_rate[res[0]] = []
            win_rate[res[0]].append(float(res[1]))
            term_rate[res[0]].append(float(res[2]))
    return iter_num, mean, maximum, start_idx, win_rate, term_rate


if __name__ == '__main__':
    iter_num, mean, maximum, start_idx, win_rate, term_rate = parse_log('300k.log')
    plt.clf()
    for target_val in ['2048', '4096', '8192', '16384']:
        try:
            complete_win_rate = complete_record(start_idx[target_val], win_rate[target_val])
            plt.plot(iter_num, complete_win_rate, label=f'win rate of {target_val}', ls='-')
        except:
            print(f'{target_val} not exist')

    plt.xlabel('Episodes')
    plt.ylabel('Rate (%)')
    plt.legend()
    plt.grid()
    plt.savefig('300k-win-rate.png', dpi=150), plt.clf()

    plt.plot(iter_num, mean, label='average score (per 1000 episodes)')
    plt.plot(iter_num, maximum, label='maximum score (per 1000 episodes)')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.savefig('300k-score.png', dpi=150), plt.clf()

    plt.clf()
    for target_val in ['2048', '4096', '8192', '16384']:
        try:
            x = list(range(start_idx[target_val], start_idx[target_val]+1000*len(term_rate[target_val]), 1000))
            plt.plot(x, term_rate[target_val], label=f'terminate with {target_val}')
        except:
            print(f'{target_val} not exist')
    plt.xlabel('Episodes')
    plt.ylabel('Rate (%)')
    plt.legend()
    plt.grid()
    plt.savefig('300k-term-rate.png', dpi=150), plt.clf()
